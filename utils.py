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
    tensor = torch.zeros(195, 195, dtype = torch.float32)
    indices = torch.tril_indices(195, 195)
    tensor[indices[0], indices[1]] = tril_tensor.to(torch.float)
    idx = np.arange(tensor.shape[0])
    tensor[idx,idx] = tensor[idx,idx]
    return tensor

def find_homo_lumo(preds):
    energies = []
    for pred in preds:
        hamiltonian = convert_tril(pred[:19110])
        
        overlap = convert_tril(pred[19110:]).fill_diagonal_(1)
        
        eigenvalues = eigvalsh(overlap.inverse() @ hamiltonian)
        
        eigenvalues_pos = eigenvalues[eigenvalues > 0]
        
        eigenvalues_neg = eigenvalues[eigenvalues < 0]
        
        HOMO = eigenvalues_neg[-5]#.max()
        
        LUMO = eigenvalues_pos[5]#.min()
                    
        energies.append([HOMO, LUMO, LUMO-HOMO])
            
    return torch.tensor(energies, requires_grad=True)

def find_homo_lumo2(preds):
    # Convert tril tensors to full tensors
    hamiltonians = [convert_tril(pred[:19110]) for pred in preds]
    overlaps = [convert_tril(pred[19110:]).fill_diagonal_(1) for pred in preds]

    overlaps = torch.stack(overlaps, dim=0)
    hamiltonians = torch.stack(hamiltonians, dim=0)

    # Compute eigenvalues
    eigenvalues = eigvalsh(overlaps.inverse() @ hamiltonians)

    first_positive_eigenvalue = [next((i for i, eigenvalue in enumerate(subtensor) if eigenvalue > 0), 0) for subtensor in eigenvalues]
    first_negative_eigenvalue = [next((i for i, eigenvalue in enumerate(subtensor) if math.isclose(eigenvalue, 0, abs_tol = 10**-1)), 0) for subtensor in eigenvalues]

    LUMO_idx = torch.tensor(first_positive_eigenvalue) + 4
    HOMO_idx = torch.tensor(first_negative_eigenvalue) - 4

    HOMO = torch.nan_to_num(eigenvalues[range(eigenvalues.shape[0]), HOMO_idx.abs()])

    # Select fifth eigenvalue above zero
    LUMO = torch.nan_to_num(eigenvalues[range(eigenvalues.shape[0]), LUMO_idx.abs()])

    # Compute HOMO-LUMO gap
    gap = LUMO - HOMO

    return torch.stack([HOMO, LUMO, gap], dim=1)