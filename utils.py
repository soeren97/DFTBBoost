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
    hamiltonian: NDArray, overlap: NDArray, n_electrons: int
) -> List:
    hamiltonian = torch.Tensor(hamiltonian)

    overlap = torch.Tensor(overlap)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlap.inverse() @ hamiltonian)

    # pad eigenvalues
    n_orbitals = len(eigenvalues)

    padded_eigenvalues = torch.zeros([65])

    padded_eigenvalues[:n_orbitals] = eigenvalues

    # find index of HOMO
    i = n_electrons // 2  # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[i - 5]

    return [HOMO, LUMO, padded_eigenvalues, n_orbitals]


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

    mask = torch.arange(65).unsqueeze(0).repeat(
        n_orbitals.shape[0], 1
    ) > n_orbitals.unsqueeze(1)

    eigenvalues = eigenvalues * mask

    # find index of HOMO
    i = n_electrons // 2  # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[range(len(eigenvalues)), i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[range(len(eigenvalues)), i - 5]

    return eigenvalues, HOMO, LUMO


def costume_collate_NN(batch):
    X = torch.stack([tensor for entry in batch for tensor in entry[0]], dim=0)

    HOMO = torch.stack([tensor for entry in batch for tensor in entry[1]], dim=0)

    LUMO = torch.stack([tensor for entry in batch for tensor in entry[2]], dim=0)

    eigenvalues = torch.stack([tensor for entry in batch for tensor in entry[3]], dim=0)

    ham_over = torch.stack([tensor for entry in batch for tensor in entry[4]], dim=0)

    n_electrons = torch.stack(
        [torch.tensor(integer) for entry in batch for integer in entry[5]], dim=0
    )

    n_orbitals = torch.stack(
        [torch.tensor(integer) for entry in batch for integer in entry[6]], dim=0
    )
    return X, HOMO, LUMO, eigenvalues, ham_over, n_electrons, n_orbitals


def costume_collate_GNN(batch):
    X = Batch.from_data_list(batch[0])

    HOMO = torch.stack(batch[1], dim=0).reshape(-1)

    LUMO = torch.stack(batch[2], dim=0).reshape(-1)

    eigenvalues = torch.stack(batch[3], dim=0).reshape(-1, 65)

    ham_over = torch.stack(batch[4], dim=0).reshape(-1, 2145 * 2)

    n_electrons = torch.stack(batch[5], dim=0).reshape(-1)

    n_orbitals = torch.stack(batch[6], dim=0).reshape(-1)

    return X, HOMO, LUMO, eigenvalues, ham_over, n_electrons, n_orbitals


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


def mixture_func(
    x: np.ndarray,
    w1: float,
    w2: float,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    p: float,
    binom_n: float,
    binom_p: float,
) -> np.ndarray:
    """
    Mixture function of a Poisson distribution and four Gaussian distributions.

    Parameters:
    x (np.ndarray): The x-values to evaluate the function at.
    w1 (float): The weight of the first Gaussian distribution.
    w2 (float): The weight of the second Gaussian distribution.
    mu1 (float): The mean of the first Gaussian distribution.
    mu2 (float): The mean of the second Gaussian distribution.
    sigma1 (float): The standard deviation of the first Gaussian distribution.
    sigma2 (float): The standard deviation of the second Gaussian distribution.
    p (float): The weight of the Poisson distribution and normalisation factor.
    binom_n (float): Binominal number of trials.
    binom_p (float): Binominal chance of succes.

    Returns:
    np.ndarray: The evaluated values of the mixture function.
    """
    gaussians = np.array(
        [
            w1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2),
            w2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2),
        ]
    )
    binom_part = p * binom.pmf(x, binom_n, binom_p)

    return np.sum(gaussians, axis=0) + binom_part
