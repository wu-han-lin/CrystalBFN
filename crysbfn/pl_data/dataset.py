import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from crysbfn.common.utils import PROJECT_ROOT
from crysbfn.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop, frac_to_cart_coords)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def _make_fully_connected_edge_index(self, n_nodes):
        device = 'cpu'
        row = (
            torch.arange(0, n_nodes, dtype=torch.long)
            .reshape(1, -1, 1)
            .repeat(1, 1, n_nodes)
            .to(device=device)
        )
        col = (
            torch.arange(0, n_nodes, dtype=torch.long)
            .reshape(1, 1, -1)
            .repeat(1, n_nodes, 1)
            .to(device=device)
        )
        full_adj = torch.cat([row, col], dim=0)
        diag_bool = torch.eye(n_nodes, dtype=torch.bool).to(device=device)
        full_adj = full_adj[:, :n_nodes, :n_nodes].reshape(2, -1)
        diag_bool = diag_bool[:n_nodes, :n_nodes].reshape(-1)
        return full_adj[:, ~diag_bool]

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        if self.scaler != None:
            prop = self.scaler.transform(data_dict[self.prop])
        else:
            # print(data_dict[self.prop])
            prop = torch.tensor(data_dict[self.prop])

        (frac_coords, atom_types, lengths, angles, edge_indices,to_jimages, num_atoms, cart_coords) = data_dict['graph_arrays']
        fully_connected_edge_index = self._make_fully_connected_edge_index(num_atoms)
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        # cart_coords = frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
        data = Data(
            frac_coords = torch.Tensor(frac_coords),
            cart_coords = torch.Tensor(cart_coords),
            atom_types  = torch.LongTensor(atom_types),
            lengths = torch.Tensor(lengths).view(1, -1),
            angles = torch.Tensor(angles).view(1, -1),
            edge_index = torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges),
            fully_connected_edge_index = torch.LongTensor(fully_connected_edge_index),
            to_jimages = torch.LongTensor(to_jimages),
            num_atoms = torch.LongTensor([num_atoms]),
            num_bonds = torch.LongTensor([edge_indices.shape[0]]),
            num_nodes = torch.LongTensor([num_atoms]),
            y=prop.view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            )

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, _ ) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords = torch.tensor(frac_coords, dtype=torch.float32),
            # cart_coords = torch.Tensor(cart_coords),
            atom_types  = torch.LongTensor(atom_types),
            lengths = torch.tensor(lengths, dtype=torch.float32).view(1, -1),
            angles = torch.tensor(angles, dtype= torch.float32).view(1, -1),
            edge_index = torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges),
            to_jimages = torch.LongTensor(to_jimages),
            num_atoms = torch.LongTensor([num_atoms]),
            num_bonds = torch.LongTensor([edge_indices.shape[0]]),
            num_nodes = torch.LongTensor([num_atoms]),
        )
        # original code
        # data = Data(
        #     frac_coords=torch.Tensor(frac_coords),
        #     atom_types=torch.LongTensor(atom_types),
        #     lengths=torch.Tensor(lengths).view(1, -1),
        #     angles=torch.Tensor(angles).view(1, -1),
        #     edge_index=torch.LongTensor(
        #         edge_indices.T).contiguous(),  # shape (2, num_edges)
        #     to_jimages=torch.LongTensor(to_jimages),
        #     num_atoms=num_atoms,
        #     num_bonds=edge_indices.shape[0],
        #     num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        # )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from crysbfn.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)
    cart_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='cart_coords')
    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    dataset.cart_scaler = cart_scaler

    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
