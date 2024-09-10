import ray
import argparse
import os
import json
import time
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from func_timeout import func_timeout, FunctionTimedOut
from cdvae.common.data_utils import Crystal, ray_crys_map
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen.analysis.local_env')
print('WARNING: ignore pymatgen UserWarning')
from collections import Counter
from eval_utils import (
    get_crystal_array_list_csp, smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov,compute_cov_sep)
import amd
from amd.io import periodicset_from_pymatgen_structure
from pymatgen.io.cif import CifBlock

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
    'qmof': {'struc': 0.2, 'comp': 4.}
}


class RecEval(object):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            # try:
            rms_dist = self.matcher.get_rms_dist(
                pred.structure, gt.structure)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
            # except Exception as e:
                # print(e)
                # return None
        validity = [c.valid for c in self.preds]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}

    def get_metrics(self):
        return self.get_match_rate_and_rms()

class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate_multi': match_rate,
                'rms_dist_multi': mean_rms_dist}    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics

class GenEval(object):
    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None, get_stability=True):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        self.valid_crys = valid_crys
        if len(valid_crys) >= n_samples: 
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            # self.valid_samples = valid_crys
            # print(f"WARNING:the number of valid generated crys {len(valid_crys)} is not enough!!!!!")
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')
        self.get_stability = get_stability
        if get_stability:
            from e_above_hull import EnergyHull
            self.energy_hull = EnergyHull()
        
        # 保存为cif string的csv文件    
        valid_cifs = [c.structure.to(fmt='cif') for c in self.valid_samples]
        import pandas as pd
        df = pd.DataFrame(valid_cifs,columns=['cif'])
        df.to_csv(f'valid_crys_{eval_model_name}.csv',index=True)

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        print('avg N_elem gen:',np.mean(pred_nelems),'avg N_elem in gt:',np.mean(gt_nelems))
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        (cov_metrics_dict_sep, combined_dist_dict) = compute_cov_sep(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        cov_metrics_dict.update(cov_metrics_dict_sep)
        return cov_metrics_dict
    
    def get_volume_wdist(self):
        pred_volume = [c.structure.volume
                       for c in self.valid_samples]
        gt_volume = [c.structure.volume for c in self.gt_crys]
        wdist_volume = wasserstein_distance(pred_volume, gt_volume)
        return {'wdist_volume': wdist_volume}

    def get_stability_rate(self):
        gen_structs = [c.structure for c in self.valid_crys]
        e_hulls = []
        meta_stable_0 = []
        meta_stable_0_1 = []
        for struct in tqdm(gen_structs,desc='computing e_hull'):
            try:
                e_hull,e_m3gnet = self.energy_hull.get_energy(struct,get_m3gnet_energy=True)
                print(e_hull)
                e_hulls.append(e_hull)
                meta_stable_0.append(int(e_hull < 0))
                meta_stable_0_1.append(int(e_hull < 0.1))
            except Exception as e:
                print('exception when computing e_hull!\n',e)
                e_hulls.append([None])
                meta_stable_0.append(0)
                meta_stable_0_1.append(0)                
        return {'meta_stable_0': np.sum(meta_stable_0)/1000,
                'meta_stable_0_1': np.sum(meta_stable_0_1)/1000,
                }

    def get_amd(self, duplicate_threshold=0.05):
        gen_structs = [periodicset_from_pymatgen_structure(c.structure) for c in tqdm(self.valid_crys,desc='converting gen structs to periodicset')]
        gt_structs = [periodicset_from_pymatgen_structure(c.structure) for c in tqdm(self.gt_crys,desc='converting gt structs to periodicset')]
        gen_amds = [amd.AMD(struct,100) for struct in tqdm(gen_structs,desc='computing AMD for gen structs')]
        gt_amds = [amd.AMD(struct,100) for struct in tqdm(gt_structs,desc='computing AMD for gt structs')]
        dm = amd.AMD_cdist(gen_amds, gt_amds)
        trainset_pairwise_amd = amd.AMD_pdist(gt_amds)
        # gen_pdds = [amd.PDD(struct,100) for struct in tqdm(gt_structs,desc='computing PPD for gen structs')]
        # gt_pdds = [amd.PDD(struct,100) for struct in tqdm(gt_structs,desc='computing PPD for gt structs')]
        # ppd_dist = amd.PDD_cdist(gen_pdds, gt_pdds)
        dm_min = dm.min(axis=1)
        percentile_threshold1 = np.percentile(trainset_pairwise_amd, [0.05])
        novel_samples1 = (dm_min > percentile_threshold1).sum()
        percentile_threshold2 = np.percentile(trainset_pairwise_amd, [0.01])
        novel_samples2 = (dm_min > percentile_threshold2).sum()
        return {'novelty_amd@0.05': novel_samples1/10000,'novelty_amd@0.01': novel_samples2/10000,'percentile@0.05':percentile_threshold1,'percentile@0.01':novel_samples2}
        
        
    def get_metrics(self,get_prop=False):
        metrics = {}
        metrics.update(self.get_amd())
        if self.get_stability:
            metrics.update(self.get_stability_rate())
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        if get_prop:
            metrics.update(self.get_prop_wdist())
        metrics.update(self.get_coverage())
        metrics.update(self.get_volume_wdist())
        return metrics


class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if len(data['frac_coords'].shape)==2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list

@ray.remote
def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 


def main(args):
    all_metrics = {}
    use_ray = False
    if ray.is_initialized():
        ray.shutdown()
    if use_ray:
        ray.init()
    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        print('transforming gen crystals!')
        if use_ray:
            gen_crys = ray.get([ray_crys_map.remote(x) for x in crys_array_list])     
        else:   
            gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if 'recon' not in args.tasks:
            gt_path = cfg.data.root_path+'/test.csv'
            csv = pd.read_csv(gt_path)
            print('transforming gt crystals!')
            gt_crys = torch.load(cfg.data.root_path+'/test_crys_list.pt')
            # gt_crys = ray.get([get_gt_crys_ori.remote(x) for x in csv['cif']])
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name, get_stability=args.stability)
        gen_metrics = gen_evaluator.get_metrics(get_prop=args.get_prop)
        all_metrics.update(gen_metrics)

    if 'csp' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'csp', args.label)
        # recon_crys_cache_path = recon_file_path + f".{'multi' if args.multi_eval else ''}_eval.cache"
        batch_idx = -1 if args.multi_eval else 0
        # if os.path.exists(recon_crys_cache_path):
        #     crys_array_list = torch.load(recon_crys_cache_path)
        # else:
        crys_array_list, true_crystal_array_list = get_crystal_array_list_csp(
            recon_file_path, batch_idx = batch_idx)
            # torch.save(crys_array_list, recon_crys_cache_path)
        gt_crys = torch.load(cfg.data.root_path+'/test_crys_list.pt')

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x,check_comp=False), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x,check_comp=False), crys_array_list[i]))   
                # if i > 3:
                #     break

        if args.multi_eval:
            rec_evaluator = RecEvalBatch(pred_crys, gt_crys[:len(pred_crys[0])])
        else:
            rec_evaluator = RecEval(pred_crys, gt_crys[:len(pred_crys)])

        recon_metrics = rec_evaluator.get_metrics()

        all_metrics.update(recon_metrics)

    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--gt_data_path',default='')
    parser.add_argument('--get_prop',action='store_true')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--stability', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
