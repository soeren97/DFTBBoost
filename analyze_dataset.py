import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import torch
import yaml

from model_handler import ModelTrainer
from models import GNN
from utils import find_eigenvalues, load_config


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

    mean_delta_all = np.mean(delta_all)
    mean_delta_HOMO_LUMO = np.mean(delta_HOMO_LUMO)
    mean_distance = np.mean(delta_Matrix)

    CONFIG["dftb_dft_delta_All"] = float(mean_delta_all)
    CONFIG["dftb_dft_delta_HOMO_LUMO"] = float(mean_delta_HOMO_LUMO)
    CONFIG["dftb_dft_delta_Matrix"] = float(mean_distance)

    with open(r"model_config/config.yaml", "w") as config_file:
        updated_file = yaml.dump(CONFIG, config_file)


def compare_datasets2(CONFIG):
    path = "Data/datasets/NN/"
    files = os.scandir(path)

    dftb_list = []
    dft_list = []
    n_electrons_list = []
    n_orbitals_list = []

    for file in files:
        data = pd.read_pickle(file.path)
        dftb_list.append(data["X"])
        dft_list = dft_list.append([row[4] for row in data["Y"]])
        n_electrons_list.append(data["N_electrons"])
        n_orbitals_list.append([row[3] for row in data["Y"]])

    dftb = torch.stack(dftb_list)
    dft = torch.stack(dft_list)
    n_electrons = np.concatenate(n_electrons_list)
    n_orbitals = np.concatenate(n_orbitals_list)


def compare_datasets3(CONFIG):
    path = f"Data/datasets/NN/"
    files = os.listdir(path)

    # Load data from all files at once
    data = np.concatenate([np.load(path + i, allow_pickle=True) for i in files])

    # Extract relevant data from the arrays
    dftb = data["X"]
    Y = data["Y"]
    n_electrons = data["N_electrons"]
    n_orbitals = Y[:, 3]
    dft = Y[:, 4]

    # Perform calculations on the arrays using numpy functions
    dftb_eigenvalues, dftb_HOMO, dftb_LUMO = find_eigenvalues(
        dftb, n_electrons, n_orbitals
    )
    dft_eigenvalues, dft_HOMO, dft_LUMO = find_eigenvalues(dft, n_electrons, n_orbitals)
    distance = np.linalg.norm(dft - dftb, axis=1)

    # Compute mean of the arrays
    mean_delta_all = np.mean(abs(dftb_eigenvalues - dft_eigenvalues))
    mean_delta_HOMO_LUMO = np.mean(
        abs(dftb_HOMO - dft_HOMO) + abs(dftb_LUMO - dft_LUMO)
    )
    mean_distance = np.mean(distance)

    CONFIG["dftb_dft_delta_All"] = float(mean_delta_all)
    CONFIG["dftb_dft_delta_HOMO_LUMO"] = float(mean_delta_HOMO_LUMO)
    CONFIG["dftb_dft_delta_All"] = float(mean_distance)

    with open(r"model_config/config.yaml", "w") as config_file:
        updated_file = yaml.dump(CONFIG, config_file)


def plot_number_of_atoms():
    path = f"Data/datasets/GNN/"
    files = os.listdir(path)
    lengths = []
    for i in tqdm(files, desc="Plot number of atoms"):
        data = pd.read_pickle(path + i)
        graphs = data["X"]
        length = [graph.num_nodes for graph in graphs]
        lengths.extend(length)
    plt.hist(lengths, bins=10)
    plt.xlabel("N_atoms")
    plt.ylabel("Frequency")
    plt.savefig("Figures/N_atoms.png")


if __name__ == "__main__":
    config = load_config()
    if None in config.values():
        compare_datasets(config)

    # plot_number_of_atoms()
