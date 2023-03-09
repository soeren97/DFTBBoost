import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import swifter
import numpy as np
import torch

from numpy.typing import NDArray
from typing import Tuple, List

from utils import load_config, extract_fock, extract_overlap, freedman_diaconis_bins

sns.set_style("white")
sns.set_style("ticks")


swifter.set_defaults(
    npartitions=8,
    dask_threshold=2,
    scheduler="processes",
    progress_bar=False,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=False,
)


class Plotter:
    def __init__(self) -> None:
        self.path = None
        self.save_dir = None

        self.goal_line = None
        self.goal_error = None

    def load_model_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        List[NDArray],
        List[NDArray],
        List[NDArray],
        List[NDArray],
        List[NDArray],
        List[NDArray],
    ]:
        if self.path == None:
            folders = os.listdir("Models/")
            folders = sorted(
                folders, key=lambda x: os.path.getmtime(os.path.join("Models", x))
            )
            self.path = f"Models/{folders[-1]}/"
        else:
            self.path = f"Models/m{self.path}/"

        loss = pd.read_pickle(self.path + "losses.pkl")
        train_pred = pd.read_pickle(self.path + "predictions/train.pkl")
        test_pred = pd.read_pickle(self.path + "predictions/test.pkl")
        valid_pred = pd.read_pickle(self.path + "predictions/valid.pkl")
        fock_preds = [
            extract_fock(pred["pred"]) for pred in [train_pred, test_pred, valid_pred]
        ]
        overlap_preds = [
            extract_overlap(pred["pred"])
            for pred in [train_pred, test_pred, valid_pred]
        ]

        fock_true = [
            extract_fock(pred["y"]) for pred in [train_pred, test_pred, valid_pred]
        ]
        overlap_true = [
            extract_overlap(pred["y"]) for pred in [train_pred, test_pred, valid_pred]
        ]

        energy_preds = [
            pred["energy_pred"] for pred in [train_pred, test_pred, valid_pred]
        ]

        energy_true = [
            pred["energy_true"] for pred in [train_pred, test_pred, valid_pred]
        ]

        return (
            loss,
            fock_preds,
            overlap_preds,
            fock_true,
            overlap_true,
            energy_preds,
            energy_true,
        )

    def plot_loss(self, loss: pd.DataFrame) -> None:
        fig, ax = plt.subplots()
        loss_train = loss.Train_loss
        loss_test = loss.Test_loss
        ax.plot(loss_train, label="Train", color="blue")
        ax.plot(loss_test, label="Test", color="orange")
        ax.axhline(
            y=self.goal_line,
            linestyle="dotted",
            alpha=0.5,
            color="black",
            label="Average dft-dftb error",
        )

        ax.axhspan(
            ymin=self.goal_line - self.goal_error,
            ymax=self.goal_line + self.goal_error,
            facecolor=".5",
            alpha=0.3,
            label="dft-dftb STD",
        )

        zoomed_ax = ax.inset_axes([0.5, 0.3, 0.4, 0.45])
        zoomed_ax.plot(loss_train[-15:], color="dodgerblue")
        zoomed_ax.plot(loss_test[-15:], color="darkorange")
        zoomed_ax.axhline(
            y=self.goal_line, linestyle="dotted", alpha=0.5, color="black"
        )

        zoomed_ax.axvspan(
            ymin=self.goal_line - self.goal_error,
            ymax=self.goal_line + self.goal_error,
            xmin=len(loss_test) - 15,
            xmax=len(loss_test) - 1,
            facecolor=".5",
            alpha=0.3,
        )

        zoomed_ax.set_ylim(bottom=0, top=3)

        ax.indicate_inset_zoom(zoomed_ax, edgecolor="grey")

        plt.legend(prop={"size": 15})
        plt.xlabel("Epoch")
        plt.ylabel("Loss [Ha] (MSE)")
        plt.tight_layout()

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.savefig(self.path + "Loss.png", dpi=600)

    def plot_energies(
        self, energies_true: NDArray, energies_pred: NDArray, data_name: str
    ) -> None:
        fig, ax = plt.subplots()
        ax.scatter(energies_pred[0], energies_true[0], linewidths=1)

        plt.xlabel("Eigen energy predictions [Ha]")
        plt.ylabel("True eigen energies [Ha]")
        plt.tight_layout()

        plt.savefig(self.path + data_name + "_energies.png", dpi=600)

    def plot_distribution(
        self, energies_true: NDArray, energies_pred: NDArray, data_name: str
    ) -> None:
        bins = freedman_diaconis_bins(energies_true[0])
        fig, ax = plt.subplots()
        ax.hist(energies_pred[0], label="Predicted eigenenergies", bins=bins)
        ax.hist(energies_true[0], label="True eigenenergies", bins=bins)
        ax.set_xlabel("Eigenenergy [Ha]")
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(self.path + f"{data_name}_energy_distribution.png", dpi=600)

    def main(self, path: str = None) -> None:
        self.path = path

        (
            loss,
            fock_preds,
            overlap_preds,
            fock_true,
            overlap_true,
            energy_preds,
            energy_true,
        ) = self.load_model_data()

        self.save_dir = self.path + "Figures/"

        os.makedirs(self.save_dir, exist_ok=True)

        config = load_config(self.path)

        self.goal_line = config[f"dftb_dft_delta_{config['loss_metric']}"]

        self.goal_error = config[f"dftb_dft_std_{config['loss_metric']}"]

        self.plot_loss(loss)

        self.plot_energies(energy_preds[1], energy_true[1], "test")

        self.plot_energies(energy_preds[2], energy_true[2], "valid")

        self.plot_distribution(energy_preds[1], energy_true[1], "test")

        self.plot_distribution(energy_preds[2], energy_true[2], "valid")


if __name__ == "__main__":
    Plotter().main()
