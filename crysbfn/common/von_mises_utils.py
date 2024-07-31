from torch.distributions import Normal
from torch.distributions.von_mises import VonMises
from torch.distributions.von_mises import _log_modified_bessel_fn as log_bessel_fn
from torch.special import i0e, i1e
import torch
import numpy as np
import ray
import crysbfn

def cosxm1(x):
    return -2 * torch.square(torch.sin(x / 2.))


def _von_mises_cdf_series(x, concentration, num_terms, dtype):
    """Computes the von Mises CDF and its derivative via series expansion."""
    # Keep the number of terms as a float. It should be a small integer, so
    # exactly representable as a float.
    num_terms = torch.tensor(num_terms, dtype=dtype).to(x.device)

    def loop_body(n, rn, drn_dconcentration, vn, dvn_dconcentration):
        """One iteration of the series loop."""
        denominator = 2. * n / concentration + rn
        ddenominator_dk = -2. * n / concentration ** 2 + drn_dconcentration
        rn = 1. / denominator
        drn_dconcentration = -ddenominator_dk / denominator ** 2

        multiplier = torch.sin(n * x) / n + vn
        vn = rn * multiplier
        dvn_dconcentration = (drn_dconcentration * multiplier +
                            rn * dvn_dconcentration)
        n = n - 1.

        return n, rn, drn_dconcentration, vn, dvn_dconcentration

    n, rn, drn_dconcentration, vn, dvn_dconcentration = (
        num_terms,
        torch.zeros_like(x).to(x.device),
        torch.zeros_like(x).to(x.device),
        torch.zeros_like(x).to(x.device),
        torch.zeros_like(x).to(x.device),
    )

    while n > 0:
        n, rn, drn_dconcentration, vn, dvn_dconcentration = loop_body(
            n, rn, drn_dconcentration, vn, dvn_dconcentration
        )
    
    cdf = .5 + x / (2. * np.pi) + vn / np.pi
    dcdf_dconcentration = dvn_dconcentration / np.pi
    # Clip the result to [0, 1].
    cdf_clipped = torch.clip(cdf, 0., 1.)
    dcdf_dconcentration = dcdf_dconcentration * ((cdf >= 0.) & (cdf <= 1.))
    
    return cdf_clipped, dcdf_dconcentration

def test():
    x = torch.arange(5, dtype=torch.float32,requires_grad=True)
    concentration = torch.arange(5, dtype=torch.float32,requires_grad=True)
    z = (np.sqrt(2. / np.pi)/ i0e(concentration)) * torch.sin(.5 * x)
    print('-'*100)
    print(i0e(concentration))
    print(concentration)

# in autograd mode, the autograd feature is disabled by default, we need to enable it
@torch.enable_grad() 
def _von_mises_cdf_normal(x, concentration, dtype):
    """Computes the von Mises CDF and its derivative via Normal approximation in PyTorch."""
    # x = sample.clone().detach().requires_grad_(True)
    # concentration = kappa.clone().detach().requires_grad_(True)
    def cdf_func(x, concentration):
        z = (np.sqrt(2. / np.pi)/ i0e(concentration)) * torch.sin(.5 * x)
        z2 = z ** 2
        z3 = z2 * z
        z4 = z2 ** 2
        c = 24. * concentration
        c1 = 56.
        xi = z - z3 / ((c - 2. * z2 - 16.) / 3. -
                        (z4 + (7. / 4.) * z2 + 167. / 2.) / (c - c1 - z2 + 3.)) ** 2
        distrib = Normal(torch.tensor(0., dtype=dtype,requires_grad=True), 
                        torch.tensor(1., dtype=dtype,requires_grad=True))
        # Using PyTorch's autograd for gradient computation
        cdf_value = distrib.cdf(xi)
        return cdf_value
    # TODO: check if this is correct
    return NotImplementedError
    cdf_value = cdf_func(x, concentration)
    cdf_value.backward(torch.ones_like(cdf_value))
    return cdf_value, concentration.grad

class VonMisesSampleWithGrad(torch.autograd.Function):
    """
    Implement the sampling function of von Mises distribution with gradient
    via Implicit Gradient.
    Translated from tensorflow_probability
    https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/distributions/von_mises.py
    """
    @staticmethod
    def _von_mises_cdf(x:torch.tensor, concentration:torch.tensor):
        '''
        Computes the cumulative density function (CDF) of von Mises distribution.
        Denote the density of vonMises(loc=0, concentration=concentration) by p(t).
        Note that p(t) is periodic, p(t) = p(t + 2 pi).
        The CDF at the point x is defined as int_{-pi}^x p(t) dt.
        Thus, when x in [-pi, pi), the CDF is in [0, 1]
        
        Args:
        x: The point at which to evaluate the CDF.
        concentration: The concentration parameter of the von Mises distribution.

        Returns:
            The value of the CDF computed elementwise.

        References:
            [1] G. Hill "Algorithm 518: Incomplete Bessel Function I_0. The Von Mises
            Distribution." ACM Transactions on Mathematical Software, 1977
        '''
         
        dtype = x.dtype
        num_periods = torch.round(x / (2. * np.pi))
        x = x - (2. * np.pi) * num_periods
        # We take the hyperparameters from Table I of [1], the row for D=8
        # decimal digits of accuracy. ck is the cut-off for concentration:
        # if concentration < ck,  the series expansion is used;
        # otherwise, the Normal approximation is used.
        ck = 10.5
        # The number of terms in the series expansion. [1] chooses it as a function
        # of concentration, n(concentration). This is hard to implement in TF.
        # Instead, we upper bound it over concentrations:
        #   num_terms = ceil ( max_{concentration <= ck} n(concentration) ).
        # The maximum is achieved for concentration = ck.
        num_terms = 20
        cdf_series, dcdf_dconcentration_series = _von_mises_cdf_series(
            x, concentration, num_terms, dtype)
        cdf_normal, dcdf_dconcentration_normal = _von_mises_cdf_normal(
            x, concentration, dtype)
        use_series = concentration < ck
        cdf = torch.where(use_series, cdf_series, cdf_normal)
        cdf = cdf + num_periods
        dcdf_dconcentration = torch.where(use_series, dcdf_dconcentration_series,
                                                        dcdf_dconcentration_normal)
        return cdf, dcdf_dconcentration
    
    @staticmethod
    def sample_standard(concentration, n_samples, epsilon = 1e-6, dtype=torch.float64, loop_patience=10000):
        '''
        batched sampling from von mises distribution via rejection sampling
        this algorithm is translated from tensorflow_probability
        https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/von_mises.py#L489
        
        :param concentration: shape (batch_size, )
        :param n_samples: the number of samples to draw
        '''
        if n_samples != 1:
            shape = (n_samples,) + concentration.shape
        else:
            shape = concentration.shape
        device = concentration.device
        concentration = torch.max(concentration, torch.tensor([epsilon], dtype=dtype).to(device))
        if torch.any(concentration > 1e14):
            return torch.randn(shape, dtype=dtype).to(device) / torch.sqrt(concentration)
        
        r = 1. + torch.sqrt(1. + 4. * concentration ** 2)
        rho = (r - torch.sqrt(2. * r)) / (2. * concentration)
        
        s_exact = (1. + rho ** 2) / (2. * rho)
        s_approimate = 1. / concentration
        
        s_concentration_cutoff_dict = {
            torch.float16: 1.8e-1,
            torch.float32: 2e-2,
            torch.float64: 1.2e-4
        }
        s_concentration_cutoff = s_concentration_cutoff_dict[dtype]
        s = torch.where(concentration > s_concentration_cutoff, s_exact, s_approimate)
        
        def loop_body(done, u, w):
            """Resample the non-accepted points."""
            u = torch.rand(shape, dtype=dtype) * 2 - 1
            u = u.to(device)
            z = torch.cos(torch.pi * u)
            w = torch.where(done, w, (1. + s * z) / (s + z))
            y = concentration * (s - w)

            v = torch.rand(shape, dtype=dtype).to(device)
            accept = (y * (2. - y) >= v) | (torch.log(y / v) + 1. >= y)

            return done | accept, u, w

        done = torch.zeros(shape, dtype=torch.bool).to(device)
        u = torch.zeros(shape, dtype=dtype).to(device)
        w = torch.zeros(shape, dtype=dtype).to(device)

        cnt = 0
        while not torch.all(done):
            cnt += 1
            if cnt > loop_patience:
                assert False, 'rejection sampling not converging!'
                break            
            done, u, w = loop_body(done, u, w)
        
        assert torch.all(w <= 1) and torch.all(w >= -1)
        standard_sample = torch.sign(u) * torch.acos(w)
        # samples = standard_sample + loc
        # samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        return standard_sample
    
    @staticmethod
    def forward(ctx, concentration, n_samples=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        samples = VonMisesSampleWithGrad.sample_standard(concentration, n_samples).requires_grad_(True)
        ctx.save_for_backward(concentration, samples)
        return samples

    @staticmethod
    def backward(ctx, dy):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        concentration, samples = ctx.saved_tensors
        cdf, dcdf_dconcentration = VonMisesSampleWithGrad._von_mises_cdf(samples, concentration)
        inv_prob = torch.exp(-concentration * cosxm1(samples)) * (
            (2. * np.pi) * i0e(concentration))  
        ret = dy * (-dcdf_dconcentration * inv_prob)
        return ret, None
    
class VonMisesHelper:
    def __init__(self, kappa1=1e3, n_steps=10, device='cuda', cache_sampling=False, **kwargs):
        self.kappa1 = torch.tensor(kappa1, dtype=torch.float64)
        self.kappa0 = torch.tensor(0.0, dtype=torch.float64)
        self.entropy0 = VonMisesHelper.entropy_wrt_kappa(self.kappa0)
        self.entropy1 = VonMisesHelper.entropy_wrt_kappa(self.kappa1)
        
        self.n_steps = n_steps
        self.device = device
        self.rsample = VonMisesSampleWithGrad.apply
        
        self.cache_sampling = cache_sampling
        if self.cache_sampling:
            self.context = ray.init(runtime_env={"py_modules": [crysbfn]},num_cpus=20,num_gpus=1)
            print('ray gpus',ray.get_gpu_ids())
            sample_alphas = kwargs['sample_alphas']
            self.num_vars = kwargs['num_vars']
            self.sample_kappas = torch.tensor(sample_alphas).unsqueeze(1).repeat(1,self.num_vars).to(self.device)
            self.cache_samp_ref = self.do_cache_sampling.remote(self)
            

    @staticmethod
    def entropy_wrt_kappa(kappa: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Compute the entropy of a von Mises distribution with respect to a given kappa.
        :param kappa: the kappa
        :return: the entropy of the von Mises distribution with respect to time
        """
        kappa  = kappa.double()
        I0e_kappa = i0e(kappa) # exp(-|kappa|)*I0(kappa)
        I1e_kappa = i1e(kappa)
        return torch.log(2 * torch.tensor(torch.pi)) + torch.log(I0e_kappa) + kappa.abs() - (kappa * I1e_kappa / I0e_kappa)
    
    def entropy_wrt_t(self, t: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Use the linear interpolation of H(1) and H(0) to compute
        the entropy of a von Mises distribution with respect to a given time.
        :param t: the time
        :return: the entropy of the von Mises distribution with respect to time
        """
        assert 0 <= t <= 1, f"t must be in [0, 1] but is {t}"
        return (1 - t) * self.entropy0 + t * self.entropy1
    
    @staticmethod
    def bayesian_update_function(m, c, y, alpha):
        '''
        Compute (m_out, c_out) = h(m, c , y, α)
        according to 
        m_out = arctan((α sin(y) + c sin(m))/( α cos(y) + c cos(m))
        c_out = sqrt(α^2 + c^2 + 2αc cos(y-m))
        :param m: the previous mean, shape (D,)
        :param c: the previous concentration, shape (D,)
        return: m_out, c_out, shape (D,)
        '''
        m_out = torch.atan2(alpha * torch.sin(y) + c * torch.sin(m), 
                            alpha * torch.cos(y) + c * torch.cos(m))
        c_out = torch.sqrt(alpha**2 + c**2 + 2 * alpha * c * torch.cos(y - m))
        return m_out, c_out
    
    @staticmethod
    def kld_von_mises(mu1, kappa1, mu2, kappa2):
        '''
        according to https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/von_mises.py#L489
        Compute the Kullback-Leibler divergence between prior and posterior
        :param mu1: the prior/known mean, shape (D,)
        :param kappa1: the prior/known concentration, shape (D,)
        :param mu2: the posterior/matching mean, shape (D,)
        :param kappa2: the posterior/matchng concentration, shape (D,)
        '''
        # first term is always zero for d = 2
        second_term = torch.log(i0e(kappa2) / i0e(kappa1)) + (kappa2 - kappa1)
        third_term = i1e(kappa1) / i0e(kappa1) * (kappa1 - kappa2 * torch.cos(mu1 - mu2)) 
        return second_term + third_term 
    
    def sample(self, loc, concentration, n_samples, 
               epsilon = 1e-6, dtype=torch.float64, 
               loop_patience=1000, device='cuda',ret_eps=False):
        '''
        :param loc: shape (batch_size, )
        :param concentration: shape (batch_size, )
        :param n_samples: the number of samples to draw
        '''
        assert loc.shape == concentration.shape
        torch_vm = VonMises(loc=loc.double(),concentration=concentration.double().clip(epsilon))
        if n_samples == 1:
            samples = torch_vm.sample().float().detach()
            samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        else:
            samples = torch_vm.sample((n_samples,)).float().detach()
            samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        return samples
        
        if n_samples != 1:
            shape = (n_samples,) + loc.shape
        else:
            shape = loc.shape
        standard_sample = self.rsample(concentration, n_samples)
        samples = standard_sample + loc
        samples = samples - 2. * torch.pi * torch.round(samples / (2. * torch.pi))
        assert not samples.isnan().any()
        if not ret_eps:
            return samples
        else:
            return samples, standard_sample
    
    @ray.remote(num_gpus=1)
    def do_cache_sampling(self,):
        return self.rsample(self.sample_kappas) # shape (n_steps, n_vars)
    
    def sample_cache(self, loc:torch.tensor, ):
        '''
        use this to replace sample to accelerate the sampling process using sample cache
        '''
        # loc shape [num_atoms, 3]
        loc_size = loc.flatten().shape[0]
        std_samples = ray.get(self.cache_samp_ref)
        cached_samples = std_samples[ : , :loc_size]
        # cached_samples shape [n_steps, num_atoms, 3]
        cached_samples = cached_samples.reshape(-1,loc.shape[0],loc.shape[1])
        x = loc.unsqueeze(0).repeat(len(self.sample_kappas),1,1)
        samples = x + cached_samples
        samples = samples - 2. * torch.pi * torch.round(
                                        samples / (2. * torch.pi))
        self.cache_samp_ref = self.do_cache_sampling.remote(self)
        return samples
        

    helper = VonMisesHelper(kappa1=10000)
    mu = torch.tensor([0.1], device='cuda')
    kappa = torch.tensor([1.0], device='cuda')
    samples = helper.sample(mu, kappa, 10000).T
    print('done!')
    import matplotlib.pyplot as plt
    plt.figure()
    print(samples)
    print(samples.shape)
    plt.hist(samples.cpu().numpy()[0,:], bins=100)
    # plt.xlim(-np.pi, np.pi)
    plt.savefig("./von_mises_samples.png")

if __name__ == '__main__':  
    # sample_test()
    # exit()
    vm = VonMisesHelper(kappa1=0)
    mu = torch.tensor(torch.ones((7,500000)),requires_grad=True, device='cuda')
    kappa = torch.tensor(np.logspace(-3,3,7)[:,None].repeat(500000,1),requires_grad=True, device='cuda')
    # kappa = torch.tensor(np.logspace(-3, 3, 7), requires_grad=True, device='cuda')
    # mu = torch.tensor([0.]*7,requires_grad=True, device='cuda')
    res = vm.sample(mu, kappa, 1)
    loss = res.sum()
    print('my defined grad')
    loss.backward()
    mu_grad = mu.grad.mean(dim=1)
    kappa_grad = kappa.grad.mean(dim=1)
    print(kappa_grad)
    print(torch.allclose(mu.grad,torch.ones_like(mu),atol=1e-1,rtol=1e-1))
    print(torch.allclose(kappa.grad,torch.zeros_like(kappa),atol=1e-1,rtol=1e-1))
    
    print('-'*5+'torch'+'-'*5)
    mu = torch.tensor([0.1],requires_grad=True)
    kappa = torch.tensor([1.0],requires_grad=True)
    pt_vm = VonMises(mu, kappa)
    res = pt_vm.sample((1,))
    print(res)
    print(mu.grad)
    print(kappa.grad)
    res.backward()
    print(mu.grad)
    print(kappa.grad)
    
    