import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import torch
import yaml
from scipy.optimize import curve_fit

from model_handler import ModelTrainer
from models import GNN
from utils import find_eigenvalues, load_config, freedman_diaconis_bins, mixture_func

from typing import Tuple


def fit_data_to_dist(data: np.ndarray) -> Tuple:
    """
    Fits data to the mixture function of a Poisson distribution and four Gaussian distributions.

    Parameters:
    data (np.ndarray): The data to fit the function to.
    num_bins (int): The number of bins to use for the histogram.

    Returns:
    tuple: A tuple containing the standard deviation of the fit, the optimized parameters, the counts in each bin,
           and the bin edges.
    """
    # Create a histogram of the sample dataset
    num_bins = freedman_diaconis_bins(data)
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the mixture function to the histogram
    initial_guess = (0.1, 0.1, 0.7, 9, 1e-3, 1e-3, 5, 0.02, 9)
    params, _ = curve_fit(mixture_func, bin_centers, hist, p0=initial_guess)

    # Compute the standard deviation of the fitted function
    std = np.mean([params[-4], params[-3], np.sqrt(params[-1])])  # fix binominal std

    return std, params, bin_centers


def compare_datasets(CONFIG):
    path = f"Data/datasets/NN/"
    files = os.listdir(path)
    delta_HOMO_LUMO = []
    delta_all = []
    delta_Matrix = []
    for i in tqdm(files, desc="Calc dft, dftb diff", leave=False):
        data = pd.read_pickle(path + i)

        dftb = torch.stack(data["X"].tolist())
        Y = data["Y"]
        n_electrons = torch.tensor(data["N_electrons"])

        n_orbitals = [row[3] for row in Y]
        dft = torch.stack([row[4] for row in Y])

        dftb_eigenvalues, dftb_HOMO, dftb_LUMO = find_eigenvalues(
            dftb, n_electrons, n_orbitals
        )
        dft_eigenvalues, dft_HOMO, dft_LUMO = find_eigenvalues(
            dft, n_electrons, n_orbitals
        )
        distance = torch.norm(dft - dftb, dim=1).numpy()

        delta_all.extend(abs(dftb_eigenvalues - dft_eigenvalues).numpy())

        delta_HOMO_LUMO.extend(
            abs(dftb_HOMO - dft_HOMO) + abs(dftb_LUMO - dft_LUMO).numpy()
        )

        delta_Matrix.extend(distance)

    data = np.array(delta_all).reshape(-1)

    num_bins = freedman_diaconis_bins(data)

    std, params, bin_centers = fit_data_to_dist(data)
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    plt.hist(data, bins=num_bins, density=True, alpha=0.5, label="Error")
    plt.plot(bin_centers, mixture_func(bin_centers, *params), "r-", label="Fit")
    plt.show()
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.ylim(top=6)
    plt.legend()
    plt.tight_layout()
    plt.xticks(np.arange(0, max(hist), 1))
    plt.savefig("Figures/error_dist.png", dpi=300)

    mean_delta_all = np.mean(delta_all)
    mean_delta_HOMO_LUMO = np.mean(delta_HOMO_LUMO)
    mean_distance = np.mean(delta_Matrix)

    std_delta_all = np.std(delta_all)
    std_delta_HOMO_LUMO = np.std(delta_HOMO_LUMO)
    std_distance = np.std(delta_Matrix)

    CONFIG["dftb_dft_delta_All"] = float(mean_delta_all)
    CONFIG["dftb_dft_delta_HOMO_LUMO"] = float(mean_delta_HOMO_LUMO)
    CONFIG["dftb_dft_delta_Matrix"] = float(mean_distance)

    CONFIG["dftb_dft_std_All"] = float(std_delta_all)
    CONFIG["dftb_dft_std_HOMO_LUMO"] = float(std_delta_HOMO_LUMO)
    CONFIG["dftb_dft_std_Matrix"] = float(std_distance)

    # with open(r"model_config/config.yaml", "w") as config_file:
    #    updated_file = yaml.dump(CONFIG, config_file)


def analyze_dataset():
    path = f"Data/datasets/GNN/"
    files = os.listdir(path)
    lengths = []
    energies = []
    matrices = []
    for i in tqdm(files, desc="Plot number of atoms"):
        data = pd.read_pickle(path + i)
        graphs = data["X"]
        length = [graph.num_nodes for graph in graphs]
        lengths.extend(length)

        matrix = [row[4] for row in data["Y"]]
        matrix = torch.stack(matrix)
        matrices.extend(matrix)

        energy = [row[2] for row in data["Y"]]
        energy = torch.stack(energy)
        energies.extend(energy)

    matrices = torch.stack(matrices).reshape(-1, 1).numpy()
    energies = torch.stack(energies).reshape(-1, 1).numpy()

    return lengths, matrices, energies


def plot_n_atoms(lengths):
    bins = freedman_diaconis_bins(lengths)
    plt.hist(lengths, bins=bins)
    plt.xlabel("N_atoms")
    plt.ylabel("Frequency")
    plt.savefig("Figures/N_atoms.png", dpi=300)
    plt.close()


def plot_energies(energies):
    bins = freedman_diaconis_bins(energies)
    plt.hist(energies, bins=bins)
    plt.xlabel("Energy [Ha]")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig("Figures/Energies.png", dpi=300)
    plt.close()


def plot_matrix_content(matrices):
    bins = freedman_diaconis_bins(matrices)
    plt.hist(matrices, bins=bins)
    plt.xlabel("Matrix component")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig("Figures/Matrix_component.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    config = load_config()
    # if None in config.values():
    compare_datasets(config)

    n_atoms, matrix_content, eigenvalues = analyze_dataset()

    plot_n_atoms(n_atoms)

    plot_energies(eigenvalues)

    plot_matrix_content(matrix_content)

    print([eigenvalues.shape, matrix_content.shape])
