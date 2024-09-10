import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from cdvae.pl_modules.vm_bfn_plmodel import VM_BFN_PL_Model
from cdvae.pl_modules.vm_bfn_csp_plmodel import VM_BFN_CSP_PL_Model
from cdvae.common.callbacks import EMACallback
from eval_utils import load_model
import warnings

# 忽略特定的警告
warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen.analysis.local_env')
print('WARNING: ignore pymatgen UserWarning')

def generation(model:VM_BFN_PL_Model, num_batches_to_sample, num_samples_per_z,
               batch_size=4, sample_steps=100, args=None):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for _ in tqdm(range(num_batches_to_sample), desc='Eval[gen]'):
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        for sample_idx in range(num_samples_per_z):
            # outputs = model.sample(z=z, num_samples=z.shape[0],sample_steps=ld_kwargs.n_step_each)
            # sample(self,  num_samples=10, sample_steps=1000, segment_ids=None, show_bar=False, **kwargs)
            samples = model.sample(num_samples=batch_size, sample_steps=sample_steps, show_bar=True, **vars(args))
            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)

def structure_prediction(loader, model:VM_BFN_CSP_PL_Model, num_evals, args=None):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []
    for idx, batch in enumerate(tqdm(loader)):
        if idx >= args.num_batches_to_csp:
                break
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []
        for eval_idx in range(num_evals):
            # print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs = model.sample(batch,show_bar=False,**vars(args))
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))

        input_data_list = input_data_list + batch.cpu().to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    input_data_batch = Batch.from_data_list(input_data_list)


    return (
        frac_coords, atom_types, lengths, angles, num_atoms, input_data_batch
    )


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    # model, test_loader, cfg = load_model(
    #     model_path, load_data=('recon' in args.tasks) or
    #     ('opt' in args.tasks and args.start_from == 'data'))
    model, test_loader, cfg = load_model(
        model_path, load_data=True)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    assert 'recon' in args.tasks \
            or 'gen' in args.tasks \
                or 'opt' in args.tasks \
                    or 'csp' in args.tasks, 'No task to evaluate.'
                
    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.n_step_each,args=args)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)
    
    if 'csp' in args.tasks:
        print('Evaluate model on the csp task.')
        start_time = time.time()
        (frac_coords, atom_types, lengths, angles, num_atoms, input_data_batch) = structure_prediction(
                test_loader, model, args.num_evals, args=args)

        if args.label == '':
            csp_out_name = 'eval_csp.pt'
        else:
            csp_out_name = f'eval_csp_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'time': time.time() - start_time,
        }, model_path / csp_out_name)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=10, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=16, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--samp_acc_factor', default=1, type=float)
    parser.add_argument('--back_sampling', action='store_true')
    parser.add_argument('--back_passes', default=1, type=int)
    parser.add_argument('--num_batches_to_csp', default=1, type=int)
    parser.add_argument('--strategy', default='end_back', type=str)
    args = parser.parse_args()
    print('label:',args.label)
    main(args)
