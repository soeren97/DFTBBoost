import torch
import numpy as np
import pandas as pd
from torch.linalg import eigvalsh
import warnings
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
import yaml

from torch.utils.data.dataset import Subset, Dataset
from typing import Sequence, Union, Generator, List, Dict


def load_config(path: str = None) -> Dict:
    """Function to load in configuration file for model.
    Used to easily change variables such as learning rate,
    loss function and such
    """
    if path == None:
        path = "model_config/config.yaml"

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


def convert_tril(tril_tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros(65, 65, dtype=torch.float32)
    indices = torch.tril_indices(65, 65)
    tensor[indices[0], indices[1]] = tril_tensor.to(torch.float)
    idx = np.arange(tensor.shape[0])
    tensor[idx, idx] = tensor[idx, idx]
    return tensor


def find_homo_lumo_pred(
    preds: List[torch.Tensor], n_electrons: List[int]
) -> torch.Tensor:
    if type(n_electrons) == torch.Tensor:
        pass
    else:
        n_electrons = torch.cat(n_electrons)

    # Convert tril tensors to full tensors
    hamiltonians = [convert_tril(pred[:2145]) for pred in preds]
    overlaps = [convert_tril(pred[2145:]).fill_diagonal_(1) for pred in preds]

    overlaps = torch.stack(overlaps, dim=0)
    hamiltonians = torch.stack(hamiltonians, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ hamiltonians)

    # find index of HOMO
    i = n_electrons // 2  # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[range(len(eigenvalues)), i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[range(len(eigenvalues)), i - 5]

    # Compute HOMO-LUMO gap
    gap = LUMO - HOMO

    return torch.stack([HOMO, LUMO, gap], dim=1)


def find_homo_lumo_true(
    hamiltonian: np.array, overlap: np.array, n_electrons: int
) -> torch.Tensor:
    overlap = torch.tensor(overlap)
    hamiltonian = torch.tensor(hamiltonian)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlap.inverse() @ hamiltonian)

    # find index of HOMO
    i = n_electrons // 2  # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[i - 5]

    # Compute HOMO-LUMO gap
    gap = LUMO - HOMO

    return torch.stack([HOMO, LUMO, gap])


def find_eigenvalues(preds: pd.DataFrame, n_electrons: List[int]) -> torch.Tensor:
    n_electrons = torch.tensor(n_electrons)

    # Convert tril tensors to full tensors
    hamiltonians = [convert_tril(pred[:2145]) for pred in preds]
    overlaps = [convert_tril(pred[2145:]).fill_diagonal_(1) for pred in preds]

    overlaps = torch.stack(overlaps, dim=0)
    hamiltonians = torch.stack(hamiltonians, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ hamiltonians)

    # find index of HOMO
    i = n_electrons // 2  # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[range(len(eigenvalues)), i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[range(len(eigenvalues)), i - 5]

    return eigenvalues, HOMO, LUMO


def costume_collate(batch):
    X = torch.stack([tensor for entry in batch for tensor in entry[0]], dim=0)
    Y = torch.stack([tensor for entry in batch for tensor in entry[1]], dim=0)
    n_electrons = torch.stack(
        [torch.tensor(integer) for entry in batch for integer in entry[2]], dim=0
    )
    energies = torch.stack([tensor for entry in batch for tensor in entry[3]], dim=0)

    return X, Y, n_electrons, energies
