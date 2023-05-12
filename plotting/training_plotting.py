import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from numpy.typing import NDArray
from typing import Tuple, List, Optional

import sys

sys.path.append(os.getcwd())

from utils import load_config, extract_fock, extract_overlap, freedman_diaconis_bins


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

        # ax.axhspan(
        #     ymin=self.goal_line - self.goal_error,
        #     ymax=self.goal_line + self.goal_error,
        #     facecolor=".5",
        #     alpha=0.3,
        #     label="dft-dftb STD",
        # )

        zoomed_ax = ax.inset_axes([0.62, 0.52, 0.35, 0.45])
        zoomed_ax.plot(loss_train[-15:], color="dodgerblue")
        zoomed_ax.plot(loss_test[-15:], color="darkorange")
        zoomed_ax.axhline(
            y=self.goal_line, linestyle="dotted", alpha=0.5, color="black"
        )

        # zoomed_ax.axvspan(
        #     ymin=self.goal_line - self.goal_error,
        #     ymax=self.goal_line + self.goal_error,
        #     xmin=len(loss_test) - 15,
        #     xmax=len(loss_test) - 1,
        #     facecolor=".5",
        #     alpha=0.3,
        # )

        zoomed_ax.set_ylim(bottom=0, top=2.5)

        ax.indicate_inset_zoom(zoomed_ax, edgecolor="grey")

        plt.legend(prop={"size": 15})
        plt.xlabel("Epoch")
        plt.ylabel("Loss [Ha] (MSE)")
        plt.tight_layout()

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.savefig(self.save_dir + "Loss.png", dpi=600)
        plt.close()

    def plot_energies(
        self,
        energies_true: pd.Series,
        energies_pred: pd.Series,
        data_name: str,
        split: bool,
    ) -> None:
        fig, ax = plt.subplots()

        # Stack energies of all molecules
        energies_true = np.hstack(energies_true[0])
        energies_pred = np.hstack(energies_pred[0])

        # Convert to eV
        # energies_true *= 27.2114
        # energies_pred *= 27.2114

        if split:
            # remove core orbitals
            mask = energies_true > -5

            energies_true = energies_true[mask]
            energies_pred = energies_pred[mask]
            name = "valence"

        else:
            name = "full"

        # Remove padding
        stacked_true = energies_true[energies_true != 0]
        stacked_pred = energies_pred[energies_pred != 0]

        # Compute density matrix
        _, xedges, yedges = np.histogram2d(stacked_pred, stacked_true, bins=200)

        im = ax.hist2d(stacked_pred, stacked_true, bins=200, cmin=1)

        # Add colorbar
        cb = fig.colorbar(im[3], ax=ax)
        cb.set_label("Count")
        save_name = f"heatmap_{data_name}_{name}.png"

        ax.plot(
            [xedges[0], xedges[-1]],
            [xedges[0], xedges[-1]],
            label="Perfect correlation",
            color="grey",
            linestyle="dashed",
        )

        ax.set_xlim(left=xedges[0], right=xedges[-1])
        ax.set_ylim(bottom=xedges[0], top=xedges[-1])

        plt.xlabel("Eigen energy predictions")
        plt.ylabel("True eigen energies")
        plt.tight_layout()
        plt.legend()
        plt.savefig(self.save_dir + save_name, dpi=600)
        plt.close()

    def plot_distribution(
        self,
        energies_true: pd.Series,
        energies_pred: pd.Series,
        data_name: str,
        split: bool,
    ) -> None:
        # Stack energies of all molecules
        energies_true = np.hstack(energies_true[0])
        energies_pred = np.hstack(energies_pred[0])

        if split:
            # devide into  orbital energies
            mask_high = -21 >= energies_true
            mask1 = mask_high

            mask_high = -16 >= energies_true
            mask_low = energies_true > -21
            mask2 = np.logical_and(mask_high, mask_low)

            mask_high = -12 >= energies_true
            mask_low = energies_true > -16
            mask3 = np.logical_and(mask_high, mask_low)

            mask_high = -5 >= energies_true
            mask_low = energies_true > -12
            mask4 = np.logical_and(mask_high, mask_low)

            mask_low = energies_true > -5
            mask5 = mask_low

            # energies_true = energies_true[mask]

            # energies_pred = energies_pred[mask]

            energy_list_true = [
                energies_true[mask1],
                energies_true[mask2],
                energies_true[mask3],
                energies_true[mask4],
                energies_true[mask5],
            ]
            energy_list_pred = [
                energies_pred[mask1],
                energies_pred[mask2],
                energies_pred[mask3],
                energies_pred[mask4],
                energies_pred[mask5],
            ]
            sections = [
                "core_lowest",
                "core_low",
                "core_high",
                "core_highest",
                "valence",
            ]

        else:
            energy_list_true = [energies_true]
            energy_list_pred = [energies_pred]
            sections = ["full"]

        for energies_pred, energies_true, section in zip(
            energy_list_pred, energy_list_true, sections
        ):
            # Remove padding
            stacked_true = energies_true[energies_true != 0]
            stacked_pred = energies_pred[energies_pred != 0]

            # bin_size = 0.8  # Set the bin size
            # bin_range = (
            #     np.min([stacked_pred, stacked_true]) - 2,
            #     np.max([stacked_pred, stacked_true]) + 2,
            # )  # Set the bin range
            # np.arange(*bin_range, bin_size)  # Create an array of bin edges

            try:
                bins = freedman_diaconis_bins(stacked_pred)
            except IndexError:
                continue

            plt.hist(
                [stacked_pred, stacked_true],
                label=["Predicted eigenenergies", "True eigenenergies"],
                bins=bins,
            )
            plt.xlabel("Eigenenergy [Ha]")
            plt.ylabel("Frequency")
            # plt.yscale("log")

            plt.legend()
            plt.tight_layout()
            plt.savefig(
                self.save_dir + f"energy_distribution_{data_name}_{section}.png",
                dpi=600,
            )
            plt.close()

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

        config = load_config(path=self.path)

        self.goal_line = config[f"dftb_dft_delta_{config['loss_metric']}"]

        self.goal_error = config[f"dftb_dft_std_{config['loss_metric']}"]

        self.plot_loss(loss)

        # Plot correlation
        self.plot_energies(energy_true[1], energy_preds[1], "test", split=False)

        self.plot_energies(energy_true[2], energy_preds[2], "valid", split=False)

        self.plot_energies(energy_true[1], energy_preds[1], "test", split=True)

        self.plot_energies(energy_true[2], energy_preds[2], "valid", split=True)

        # Plot histogram of energies
        self.plot_distribution(energy_true[1], energy_preds[1], "test", split=True)

        self.plot_distribution(energy_true[1], energy_preds[1], "test", split=False)

        self.plot_distribution(energy_true[2], energy_preds[2], "valid", split=True)

        self.plot_distribution(energy_true[2], energy_preds[2], "valid", split=False)


if __name__ == "__main__":
    Plotter().main("1681562523")
