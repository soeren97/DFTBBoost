""" Dataset used to load molecules."""
import glob
from typing import Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class CostumDataset(Dataset):
    """Class used to load data.

    Args:
        Dataset (Dataset): Abstract class representing a Dataset.
    """

    def __init__(self, ml_method: str) -> None:
        """Initializes the dataset.

        Uses the machine learning method to find data location.

        Args:
            ml_method (str): Method used for training or analysis.
        """
        self.data_location = "Data/datasets/" + ml_method + "/"
        self.file_names = glob.glob(self.data_location + "*.pkl")
        self.ml_method = ml_method

    def __len__(self) -> int:
        """Return the number of pickle files in the data location.

        Each pickle file contains up to 32 molecules.

        Returns:
            int: Number of pickle files in the data location.
        """
        return len(self.file_names)

    def __getitem__(self, index: int) -> List[Any]:
        """Return a list of properties from the molecules in a pickle file.

        Args:
            index (int): Index number of the pickle file to be loaded.

        Returns:
            List[Any]: List of proberties for the up to 32 molecules.
        """
        data = pd.read_pickle(self.file_names[index])
        if self.ml_method != "NN":
            X = data["X"].tolist()

            Y = data["Y"]

            eigenvalues = [row[0] for row in Y]

            N_orbitals = [row[1] for row in Y]

            ham_over = [row[2] for row in Y]

            N_electrons = data["N_electrons"].tolist()

            return [X, eigenvalues, ham_over, N_electrons, N_orbitals]

        else:
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
