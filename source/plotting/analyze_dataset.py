""" Investigate the structure and proberties of the molecules in the dataset."""
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from tqdm import tqdm

# from source.models.model_handler import ModelTrainer
from source.models.models import GNN  # noqa: F401
from source.utils import (
    combine_distributions,
    find_eigenvalues,
    fit_histogram_peaks,
    freedman_diaconis_bins,
    load_config,
)


def fit_data_to_dist(data: np.ndarray, num_bins: int) -> Tuple:
    """
    Fit data to the mixture functions.

    Parameters:
    data (np.ndarray): The data to fit the function to.
    num_bins (int): The number of bins to use for the histogram.

    Returns:
    tuple: A tuple containing the standard deviation of the fit, the optimized
    parameters, the counts in each bin and the bin edges.
    """
    # Create a histogram of the sample dataset
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # fit gaussian functions to histogram
    gaus_fits = fit_histogram_peaks(hist, bin_edges)

    # Fit the mixture function to the histogram
    initial_guess = np.array(gaus_fits[0] + (9, 1e-3, 9), dtype="float64")
    params, _ = curve_fit(combine_distributions, bin_centers, hist, p0=initial_guess)

    # Compute the standard deviation of the fitted function
    std = np.sqrt(params[0] * params[1] * (1 - params[1]) * params[2])

    return std, params, bin_centers


def compare_datasets(CONFIG):
    path = "Data/datasets/NN/"
    files = os.listdir(path)
    delta_all = []
    delta_Matrix = []
    for i in tqdm(files, desc="Calc dft, dftb diff", leave=False):
        data = pd.read_pickle(path + i)

        dftb = torch.stack(data["X"].tolist())
        Y = data["Y"]
        n_electrons = torch.tensor(data["N_electrons"])

        n_orbitals = [row[1] for row in Y]
        dft = torch.stack([row[2] for row in Y])

        dftb_eigenvalues = find_eigenvalues(dftb, n_electrons, n_orbitals)
        dft_eigenvalues = find_eigenvalues(dft, n_electrons, n_orbitals)
        distance = torch.norm(dft - dftb, dim=1).numpy()

        delta_all.extend(abs(dftb_eigenvalues - dft_eigenvalues).numpy())

        delta_Matrix.extend(distance)

    data = np.array(delta_all).reshape(-1)

    num_bins = int(freedman_diaconis_bins(data) / 2)

    std, params, bin_centers = fit_data_to_dist(data, num_bins)
    hist, bin_edges = np.histogram(data, bins=num_bins, density=True)
    plt.hist(data, bins=num_bins, density=True, alpha=0.5, label="Error")
    plt.plot(
        bin_centers, combine_distributions(bin_centers, *params), "r-", label="Fit"
    )
    plt.show()
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.xlim(left=0)
    plt.legend()
    plt.xticks(np.arange(0, max(hist), 1))
    plt.tight_layout()
    plt.savefig("Figures/error_dist.png", dpi=300)

    mean_delta_all = np.mean(delta_all)
    mean_distance = np.mean(delta_Matrix)

    std_delta_all = np.std(delta_all)
    std_distance = np.std(delta_Matrix)

    CONFIG["dftb_dft_delta_All"] = float(mean_delta_all)
    CONFIG["dftb_dft_delta_Matrix"] = float(mean_distance)

    CONFIG["dftb_dft_std_All"] = float(std_delta_all)
    CONFIG["dftb_dft_std_Matrix"] = float(std_distance)

    # with open(r"model_config/config.yaml", "w") as config_file:
    #    updated_file = yaml.dump(CONFIG, config_file)


def extract_n_non_h_atoms(data: pd.DataFrame) -> List[int]:
    graphs = data["X"]
    n_atom = np.array(
        [
            len(graph.node_stores[0]["x"][graph.node_stores[0]["x"] != 0])
            for graph in graphs
        ]
    )

    n_c = np.array(
        [
            len(graph.node_stores[0]["x"][graph.node_stores[0]["x"] == 1])
            for graph in graphs
        ]
    )

    n_n = np.array(
        [
            len(graph.node_stores[0]["x"][graph.node_stores[0]["x"] == 2])
            for graph in graphs
        ]
    )

    n_o = np.array(
        [
            len(graph.node_stores[0]["x"][graph.node_stores[0]["x"] == 3])
            for graph in graphs
        ]
    )

    n_f = np.array(
        [
            len(graph.node_stores[0]["x"][graph.node_stores[0]["x"] == 4])
            for graph in graphs
        ]
    )

    n_c = list(n_c[n_c != 0])

    n_n = list(n_n[n_n != 0])

    n_o = list(n_o[n_o != 0])

    n_f = list(n_f[n_f != 0])

    return n_atom, n_c, n_n, n_o, n_f


def analyze_dataset() -> Tuple[List, NDArray, NDArray]:
    path = "Data/datasets/GNN/"
    files = os.listdir(path)
    n_atoms = []
    n_carbon = []
    n_nitrogen = []
    n_oxygen = []
    n_flour = []
    energies = []
    matrices = []
    for i in tqdm(files, desc="Extract graph data"):
        data = pd.read_pickle(path + i)

        # extract number of non hydrogen atoms
        n_atom, n_c, n_n, n_o, n_f = extract_n_non_h_atoms(data)

        n_atoms.extend(n_atom)
        n_carbon.extend(n_c)
        n_nitrogen.extend(n_n)
        n_oxygen.extend(n_o)
        n_flour.extend(n_f)

        matrix = [row[2] for row in data["Y"]]
        matrix = torch.stack(matrix)
        matrices.extend(matrix)

        energy = [row[0] for row in data["Y"]]
        energy = torch.stack(energy)
        energies.extend(energy)

    matrices = torch.stack(matrices).reshape(-1, 1).numpy()
    energies = torch.stack(energies).reshape(-1, 1).numpy()

    energies = energies[energies != 0]
    n_atoms_list = [n_atoms, n_carbon, n_nitrogen, n_oxygen, n_flour]

    return n_atoms_list, matrices, energies


def plot_n_atoms(lengths: List[int], name: str) -> None:
    bins = np.arange(0.5, 9.5, 1)
    plt.hist(lengths, bins=bins, width=0.8)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig(f"Figures/{name}.png", dpi=300)
    plt.close()


def plot_elements(
    n_c: List[int], n_n: List[int], n_o: List[int], n_f: List[int]
) -> None:
    bins = np.arange(0.5, 9.5, 1)
    plt.hist(
        [n_c, n_n, n_o, n_f],
        bins=bins,
        width=0.2,
        label=[
            "Number of Carbons",
            "Number of Nitrogens",
            "Number of Oxygens",
            "Number of Fluorines",
        ],
    )
    plt.xlabel("Number of Elements")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Figures/n_elements.png", dpi=300)
    plt.close()


def plot_energies(energies: NDArray, valence: bool) -> None:
    if valence:
        # remove core orbitals
        mask = energies > -1
        energies = energies[mask]

        path = "Figures/Energies_valence.png"
    else:
        path = "Figures/Energies.png"

    bins = freedman_diaconis_bins(energies)
    plt.hist(energies, bins=bins)
    plt.xlabel("Energy [Ha]")
    plt.ylabel("Frequency")
    plt.savefig(path, dpi=300)
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
    # compare_datasets(config)

    n_atoms, matrix_content, eigenvalues = analyze_dataset()

    plot_n_atoms(n_atoms[0], "Number of Atoms")

    plot_elements(n_atoms[1], n_atoms[2], n_atoms[3], n_atoms[4])

    plot_energies(eigenvalues, valence=False)
    plot_energies(eigenvalues, valence=True)

    plot_matrix_content(matrix_content)

    print([eigenvalues.shape, matrix_content.shape])
