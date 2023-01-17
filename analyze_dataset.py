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
    for i in tqdm(files, desc="Calc dft, dftb diff", leave=False):
        data = pd.read_pickle(path + i)

        n_electrons = data["N_electrons"].to_list()
        dftb = data["NN_X"]
        dft = data["NN_Y"]

        dftb_eigenvalues, dftb_HOMO, dftb_LUMO = find_eigenvalues(dftb, n_electrons)
        dft_eigenvalues, dft_HOMO, dft_LUMO = find_eigenvalues(dft, n_electrons)

        delta_all.append(np.average(abs(dftb_eigenvalues - dft_eigenvalues)))
        delta_HOMO_LUMO.append(
            np.average(abs(dftb_HOMO - dft_HOMO) + abs(dftb_LUMO - dft_LUMO))
        )

    mean_delta_all = np.mean(delta_all)
    mean_delta_HOMO_LUMO = np.mean(delta_HOMO_LUMO)

    CONFIG["dftb_dft_delta_all"] = float(mean_delta_all)
    CONFIG["dftb_dft_delta_HOMO_LUMO"] = float(mean_delta_HOMO_LUMO)

    with open(r"model_config/config.yaml", "w") as config_file:
        updated_file = yaml.dump(CONFIG, config_file)


def plot_number_of_atoms():
    path = f"Data/datasets/GNN/"
    files = os.listdir(path)
    lengths = []
    for i in tqdm(files, desc="Plot number of atoms"):
        data = pd.read_pickle(path + i)
        graphs = data["GNN"]
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

    plot_number_of_atoms()
