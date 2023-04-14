import glob

import pandas as pd
from typing import List, Any

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

    def __getitem__(self, index: int) -> List[Any]:
        if self.ml_method != "NN":
            data = pd.read_pickle(self.file_names[index])

            X = data["X"].tolist()

            Y = data["Y"]

            eigenvalues = [row[0] for row in Y]

            N_orbitals = [row[1] for row in Y]

            ham_over = [row[2] for row in Y]

            N_electrons = data["N_electrons"].tolist()

            return [X, eigenvalues, ham_over, N_electrons, N_orbitals]

        else:
            data = pd.read_pickle(self.file_names[index])

            X = torch.stack(data["X"].tolist())

            Y = data["Y"]

            eigenvalues = torch.stack([row[0] for row in Y])

            N_orbitals = torch.stack([torch.tensor(row[1]) for row in Y])

            ham_over = torch.stack([row[2] for row in Y])

            N_electrons = torch.stack(data["N_electrons"].apply(torch.tensor).tolist())

            return {
                "X": X,
                "eigenvalues": eigenvalues,
                "ham_over": ham_over,
                "N_electrons": N_electrons,
                "N_orbitals": N_orbitals,
            }
