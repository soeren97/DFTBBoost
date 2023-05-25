"""Various functions used different places."""
import math
import os
import warnings
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import binom, norm
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.linalg import eigvalsh
from torch.utils.data.dataset import Dataset, Subset
from torch_geometric.data import Batch


def load_config(
    path: Optional[str] = None, model_name: Optional[str] = "config"
) -> Dict:
    """Function to load in configuration file for model.

    Used to easily change variables such as learning rate,
    loss function and such.

    Args:
        path (Optional[str], optional): path to config file. Defaults to None.
        model_name (Optional[str], optional): Name of model. Defaults to "config".

    Returns:
        Dict: Model config.
    """
    if path is None:
        path = f"model_config/{model_name}.yaml"
    else:
        path = f"{path}config.yaml"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_savefolder(folder_name: str) -> None:
    """Create a safe folder if it does not exist.

    Args:
        folder_name (str): Name of fodler
    """
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

    This function is a copy of the currently used torch.utils.data.random_split().

    Args:
        dataset (Dataset[torch.Tensor]): Dataset to be split.
        lengths (Sequence[Union[int, float]]): Lengths of the split.
        generator (Generator, optional): Generator. Defaults to default_generator.

    Raises:
        ValueError: Length of dataset is zero
        ValueError: Cannot verify that dataset is Sized

    Returns:
        List[Subset[torch.Tensor]]: Split dataset.
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

    # type: ignore[call-overload]
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def convert_tril(tril_tensor: torch.Tensor, n_orbitals: torch.Tensor) -> torch.Tensor:
    """Convert triangle array to matrix.

    Args:
        tril_tensor (torch.Tensor): Triangle array.
        n_orbitals (torch.Tensor): Number of orbitals.

    Returns:
        torch.Tensor: Matrix.
    """
    tensor = torch.zeros(65, 65, dtype=torch.float32)
    indices = torch.tril_indices(65, 65)
    tensor[indices[0], indices[1]] = tril_tensor.to(torch.float)
    tensor[n_orbitals:] *= 0
    tensor[:, n_orbitals:] *= 0
    return tensor


def find_eigenvalues_true(fock: NDArray, overlap: NDArray, n_orbitals: int) -> List:
    """Calculate true eigenvales.

    Args:
        fock (NDArray): Fock matrix.
        overlap (NDArray): Overlap matrix.
        n_orbitals (int): Number of orbitals.

    Returns:
        List: Eigenvalues.
    """
    # Convert tril tensors to full tensors
    fock = convert_tril(fock, n_orbitals)

    overlap = convert_tril(overlap, n_orbitals).fill_diagonal_(1)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlap.inverse() @ fock)

    return [eigenvalues, n_orbitals]


def find_eigenvalues(
    preds: torch.Tensor, n_electrons: torch.Tensor, n_orbitals: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate predicted eigenvalues.

    Args:
        preds (torch.Tensor): Predictions
        n_electrons (torch.Tensor): Number of electrons
        n_orbitals (List[torch.Tensor]): Number of orbitals.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Eigenvalues.
    """
    n_electrons = n_electrons.numpy()

    # Convert tril tensors to full tensors
    focks = [
        convert_tril(pred[:2145], n_orbital)
        for pred, n_orbital in zip(preds, n_orbitals)
    ]

    overlaps = [
        convert_tril(pred[2145:], n_orbital).fill_diagonal_(1)
        for pred, n_orbital in zip(preds, n_orbitals)
    ]

    overlaps = torch.stack(overlaps, dim=0)
    focks = torch.stack(focks, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ focks)

    return eigenvalues


def costume_collate_NN(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
    """Collate function to help load in batch data for the NN model.

    Args:
        batch (Dict[str, torch.Tensor]): Batch in need of transformation.

    Returns:
        Tuple[torch.Tensor]: Molecule properties.
    """
    X = torch.cat([d["X"] for d in batch]).float()

    eigenvalues = torch.cat([d["eigenvalues"] for d in batch])

    ham_over = torch.cat([d["ham_over"] for d in batch])

    n_electrons = torch.cat([d["N_electrons"] for d in batch])

    n_orbitals = torch.cat([d["N_orbitals"] for d in batch])

    return X, eigenvalues, ham_over, n_electrons, n_orbitals


def costume_collate_GNN(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
    """Collate function to help load in batch data for the graph models.

    Args:
        batch (Dict[str, torch.Tensor]): Batch in need of transformation.

    Returns:
        Tuple[torch.Tensor]: Molecule properties.
    """
    X = Batch.from_data_list(batch[0])

    eigenvalues = torch.stack(batch[1], dim=0).reshape(-1, 65)

    ham_over = torch.stack(batch[2], dim=0).reshape(-1, 2145 * 2)

    n_electrons = torch.stack(batch[3], dim=0).reshape(-1)

    n_orbitals = torch.stack(batch[4], dim=0).reshape(-1)

    return X, eigenvalues, ham_over, n_electrons, n_orbitals


def extract_fock(matrix: NDArray) -> NDArray:
    """Extract Fock matrix from concatenated overlap and Fock array.

    Args:
        matrix (NDArray): Concatenated overlap and Fock array.

    Returns:
        NDArray: Fock array.
    """
    fock_matrix = matrix[:2145]
    return np.array(fock_matrix)


def extract_overlap(matrix: NDArray) -> NDArray:
    """Extract overlap matrix from concatenated overlap and Fock array.

    Args:
        matrix (NDArray): Concatenated overlap and Fock array.

    Returns:
        NDArray: Overlap array.
    """
    overlap_matrix = matrix[2145:]
    return np.array(overlap_matrix)


def freedman_diaconis_bins(data: NDArray) -> int:
    """Calculate the right number of bins for a histogram.

    Args:
        data (NDArray): Data.

    Returns:
        int: Number of bins.
    """
    # Calculate the interquartile range (IQR) of the data
    iqr = np.subtract(*np.percentile(data, [75, 25]))

    # Calculate the bin width using the Freedman-Diaconis rule
    bin_width = 2 * iqr * (len(data) ** (-1 / 3))

    # Calculate the number of bins
    n_bins = (np.max(data) - np.min(data)) / bin_width

    return int(n_bins)


def gaussian(x: NDArray, a: float, x0: float, sigma: float) -> NDArray:
    """
    Calculate the value of a Gaussian function at given points.

    Args:
        x: Input array of points.
        a: Amplitude of the Gaussian function.
        x0: Mean (center) of the Gaussian function.
        sigma: Standard deviation of the Gaussian function.

    Returns:
        NDArray: Array of values representing the Gaussian function at each point.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def fit_histogram_peaks(
    hist: NDArray, bins: NDArray
) -> list[tuple[float, float, float]]:
    """Find peaks in a histogram and fits them with Gaussian distributions.

    Args:
        hist: Array of histogram values.
        bins: Array of bin edges for the histogram.

    Returns:
        List of tuples representing the fitted Gaussian distributions.
        Tuple contains: A scaling constant, a mean, and a standard deviation.
    """
    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, height=[0.1, 1])

    # Fit each peak with a Gaussian distribution
    fits = []
    for peak in peaks:
        x = (bins[peak] + bins[peak + 1]) / 2  # Use bin center as mean guess
        y = hist[peak]
        p0 = [y, x, np.std(bins)]  # Initial guess for scaling, mean, and std
        try:
            popt, _ = curve_fit(gaussian, bins[:-1], hist, p0=p0)
            fits.append(tuple(popt))
        except RuntimeError:
            continue

    return fits


def combine_distributions(x: NDArray, *args: tuple[float, float, float]) -> NDArray:
    """Combine Gaussian and binomial distributions.

    Args:
        *args: Variable number of tuples representing Gaussian distributions.
            Each tuple contains: a scaling factor, a mean, and a standard deviation.

    Returns:
        NDArray: Probability mass function for the combined distribution.
    """
    A = args[-3]
    n = args[-2]
    p = args[-1]
    gaussians = args[:-3]
    pmf = np.zeros(len(x))
    for i in range(int(len(gaussians) / 3)):
        scale = gaussians[i * 3]
        mean = gaussians[i * 3 + 1]
        std = gaussians[i * 3 + 2]
        pmf += scale * norm.pdf(x, loc=mean, scale=std)
    pmf /= np.sum(pmf)
    pmf *= A * binom.pmf(x, n, p)
    return pmf
