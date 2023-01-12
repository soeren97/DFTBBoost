import glob

import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class CostumDataset(Dataset):
    def __init__(self, ml_method: str) -> None:
        self.data_location = "Data/datasets/" + ml_method + "/"
        self.file_names = glob.glob(self.data_location + "*.pkl")
        self.ml_method = ml_method

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple:
        data = pd.read_pickle(self.file_names[index])

        if self.ml_method in ["GNN", "GNN_plus"]:
            return data["N_electrons"].tolist(), data[self.ml_method].tolist()

        else:
            X = data[f"{self.ml_method}_X"].tolist()

            Y = data[f"{self.ml_method}_Y"].tolist()

            N_electrons = data["N_electrons"].tolist()

            energies = data["Energies"].tolist()

            return X, Y, N_electrons, energies
