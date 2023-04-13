import torch
import numpy as np
import pandas as pd
from torch.linalg import eigvalsh
import warnings
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
import yaml
from scipy.stats import binom
import os

from torch.utils.data.dataset import Subset, Dataset
from torch_geometric.data import Batch

from typing import Sequence, Union, Generator, List, Dict, Tuple, Optional
from numpy.typing import NDArray


def load_config(
    path: Optional[str] = None, model_name: Optional[str] = "config"
) -> Dict:
    """Function to load in configuration file for model.

    Used to easily change variables such as learning rate,
    loss function and such
    """
    if path == None:
        path = f"model_config/{model_name}.yaml"
    else:
        path = f"{path}config.yaml"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_savefolder(folder_name):
    if os.path.isdir(folder_name):
        return
    else:
        os.makedirs(folder_name)


def random_split(
    dataset: Dataset[torch.Tensor],
    lengths: Sequence[Union[int, float]],
    generator: Generator = default_generator,
) -> List[Subset[torch.Tensor]]:
    """As an older version of torch is used random split was not implimented.
    This function is a copy of the currently implimented torch.utils.data.random_split().
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def convert_tril(tril_tensor: torch.Tensor, n_orbitals: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros(65, 65, dtype=torch.float32)
    indices = torch.tril_indices(65, 65)
    tensor[indices[0], indices[1]] = tril_tensor.to(torch.float)
    tensor[n_orbitals:] *= 0
    tensor[:, n_orbitals:] *= 0
    return tensor


def find_eigenvalues_true(
    hamiltonian: NDArray, overlap: NDArray, n_orbitals: int
) -> List:
    # Convert tril tensors to full tensors
    hamiltonian = convert_tril(hamiltonian, n_orbitals)

    overlap = convert_tril(overlap, n_orbitals).fill_diagonal_(1)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlap.inverse() @ hamiltonian)

    return [eigenvalues, n_orbitals]


def find_eigenvalues(
    preds: torch.Tensor, n_electrons: torch.Tensor, n_orbitals: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_electrons = n_electrons.numpy()

    # Convert tril tensors to full tensors
    hamiltonians = [
        convert_tril(pred[:2145], n_orbital)
        for pred, n_orbital in zip(preds, n_orbitals)
    ]

    overlaps = [
        convert_tril(pred[2145:], n_orbital).fill_diagonal_(1)
        for pred, n_orbital in zip(preds, n_orbitals)
    ]

    overlaps = torch.stack(overlaps, dim=0)
    hamiltonians = torch.stack(hamiltonians, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ hamiltonians)

    return eigenvalues


def costume_collate_NN(batch):
    X = torch.stack(batch[0]).reshape(-1, 2145)

    eigenvalues = torch.stack(batch[1], dim=0).reshape(-1, 65)

    ham_over = torch.stack(batch[2], dim=0).reshape(-1, 2145 * 2)

    n_electrons = torch.stack(batch[3], dim=0).reshape(-1)

    n_orbitals = torch.stack(batch[4], dim=0).reshape(-1)

    return X, eigenvalues, ham_over, n_electrons, n_orbitals


def costume_collate_GNN(batch):
    X = Batch.from_data_list(batch[0])

    eigenvalues = torch.stack(batch[1], dim=0).reshape(-1, 65)

    ham_over = torch.stack(batch[2], dim=0).reshape(-1, 2145 * 2)

    n_electrons = torch.stack(batch[3], dim=0).reshape(-1)

    n_orbitals = torch.stack(batch[4], dim=0).reshape(-1)

    return X, eigenvalues, ham_over, n_electrons, n_orbitals


def extract_fock(matrix: NDArray) -> NDArray:
    hf_matrix = matrix[:2145]
    return np.array(hf_matrix)


def extract_overlap(matrix: NDArray) -> NDArray:
    overlap_matrix = matrix[2145:]
    return np.array(overlap_matrix)


def calculate_transmission(
    fock_matrix: NDArray, overlap_matrix: NDArray, energy_range: NDArray
) -> float:
    # Calculate Hamiltonian matrix
    hamiltonian_matrix = fock_matrix + overlap_matrix

    # Calculate Green's function
    greens_function = np.linalg.inv(
        energy_range * np.identity(len(hamiltonian_matrix)) - hamiltonian_matrix
    )

    # Calculate transmission coefficient
    left_matrix = np.matmul(
        np.matmul(
            np.conj(
                np.transpose(
                    greens_function[: len(overlap_matrix), : len(overlap_matrix)]
                )
            ),
            overlap_matrix,
        ),
        greens_function[: len(overlap_matrix), : len(overlap_matrix)],
    )
    right_matrix = np.matmul(
        np.matmul(
            np.conj(
                np.transpose(
                    greens_function[-len(overlap_matrix) :, -len(overlap_matrix) :]
                )
            ),
            overlap_matrix,
        ),
        greens_function[-len(overlap_matrix) :, -len(overlap_matrix) :],
    )
    transmission = np.trace(np.matmul(left_matrix, np.conj(np.transpose(right_matrix))))

    return transmission.real


def freedman_diaconis_bins(data):
    # Calculate the interquartile range (IQR) of the data
    iqr = np.subtract(*np.percentile(data, [75, 25]))

    # Calculate the bin width using the Freedman-Diaconis rule
    bin_width = 2 * iqr * (len(data) ** (-1 / 3))

    # Calculate the number of bins
    n_bins = (np.max(data) - np.min(data)) / bin_width

    return int(n_bins)


def binominal_dist(
    x: np.ndarray,
    binom_n: float,
    binom_p: float,
    A: float
) -> np.ndarray:
    """
    Binoninal distribution.

    Parameters:
    x (float): Number of succeses.
    binom_n (float): Binominal number of trials.
    binom_p (float): Binominal chance of succes.
    A (float): Normalization factor.

    Returns:
    np.ndarray: The evaluated values of the mixture function.
    """
    
    binominal = A * binom.pmf(x, binom_n, binom_p)

    return binominal
