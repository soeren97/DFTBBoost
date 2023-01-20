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

        X = data["X"].tolist()

        Y = data["Y"]

        HOMO = [row[0] for row in Y]

        LUMO = [row[1] for row in Y]

        eigenvalues = [row[2] for row in Y]

        N_orbitals = [row[3] for row in Y]

        ham_over = [row[4] for row in Y]

        N_electrons = data["N_electrons"].tolist()

        return [X, HOMO, LUMO, eigenvalues, ham_over, N_electrons, N_orbitals]
