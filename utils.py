import torch
import numpy as np
from torch.linalg import eigvalsh
import warnings
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import yaml

def load_config():
    """Function to load in configuration file for model. 
    Used to easily change variables such as learning rate, 
    loss function and such
    """        
    location = 'model_config/config.yaml'
    with open(location, 'r') as file:
        config = yaml.safe_load(file)
    return config

def random_split(dataset, lengths,
                 generator=default_generator):
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
                warnings.warn(f"Length of split at index {i} is 0. "
                            f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def convert_tril(tril_tensor):
    tensor = torch.zeros(65, 65, dtype = torch.float32)
    indices = torch.tril_indices(65, 65)
    tensor[indices[0], indices[1]] = tril_tensor.to(torch.float)
    idx = np.arange(tensor.shape[0])
    tensor[idx,idx] = tensor[idx,idx]
    return tensor
    
def find_homo_lumo_pred(preds, n_electrons):    
    n_electrons = torch.cat(n_electrons)
    # Convert tril tensors to full tensors
    hamiltonians = [convert_tril(pred[:2145]) for pred in preds]
    overlaps = [convert_tril(pred[2145:]).fill_diagonal_(1) for pred in preds]

    overlaps = torch.stack(overlaps, dim=0)
    hamiltonians = torch.stack(hamiltonians, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ hamiltonians)

    # find index of HOMO
    i = n_electrons // 2 # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[range(len(eigenvalues)), i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[range(len(eigenvalues)), i - 5]

    # Compute HOMO-LUMO gap
    gap = LUMO - HOMO

    return torch.stack([HOMO, LUMO, gap], dim=1)

def find_homo_lumo_true(hamiltonian, overlap, n_electrons):
    overlap = torch.tensor(overlap)
    hamiltonian = torch.tensor(hamiltonian)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlap.inverse() @ hamiltonian)

    # find index of HOMO
    i = n_electrons // 2 # fix?

    # Select fifth eigenvalue above LUMO
    LUMO = eigenvalues[i + 6]

    # Select fifth eigenvalue bellow HOMO
    HOMO = eigenvalues[i - 5]

    # Compute HOMO-LUMO gap
    gap = LUMO - HOMO

    return torch.stack([HOMO, LUMO, gap])
