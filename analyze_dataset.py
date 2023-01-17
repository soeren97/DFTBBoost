import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import torch

from model_handler import ModelTrainer
from models import GNN
from utils import find_eigenvalues


def compare_datasets():
    path = f"Data/datasets/NN/"
    files = os.listdir(path)
    delta = []
    for i in tqdm(files, desc="Plot dft, dftb diff"):
        data = pd.read_pickle(path + i)
        dftb = data["NN_X"]
        dft = data["NN_Y"]
        dftb_eigenvalues = find_eigenvalues(dftb)
        dft_eigenvalues = find_eigenvalues(dft)
        delta.append(torch.mean(abs(dftb_eigenvalues - dft_eigenvalues)))
    return torch.mean(delta)


def plot_number_of_atoms(model):
    path = f"Data/datasets/GNN/"
    files = os.listdir(path)
    lengths = []
    for i in tqdm(files, desc="Plot number of atoms"):
        data = pd.read_pickle(path + i)
        graphs = data["GNN"]
        length = [graph.num_nodes for graph in graphs]
        lengths.extend(length)
    plt.hist(lengths, bins=10)
    plt.x_label("N_atoms")
    plt.y_label("Frequency")
    plt.savefig("Figures/N_atoms.jpg")


if __name__ == "__main__":
    difference = compare_datasets()
    plot_number_of_atoms()
