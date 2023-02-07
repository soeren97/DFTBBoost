import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_config

sns.set_style("white")
sns.set_style("ticks")


class Plotter:
    def __init__(self) -> None:
        self.path = None
        self.save = False
        self.goal_line = None

    def load_model_data(self) -> pd.DataFrame:
        if self.path == None:
            folders = os.listdir("Models/")
            folders = sorted(
                folders, key=lambda x: os.path.getmtime(os.path.join("Models", x))
            )
            self.path = "Models/" + folders[-1] + "/"
        return pd.read_pickle(self.path + "losses.pkl")

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

        zoomed_ax = ax.inset_axes([0.5, 0.3, 0.4, 0.45])
        zoomed_ax.plot(loss_train[-15:], color="dodgerblue")
        zoomed_ax.plot(loss_test[-15:], color="darkorange")
        zoomed_ax.axhline(
            y=self.goal_line, linestyle="dotted", alpha=0.5, color="black"
        )
        zoomed_ax.set_ylim(bottom=0)

        ax.indicate_inset_zoom(zoomed_ax, edgecolor="grey")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [Ha] (MSE)")
        plt.tight_layout()

        plt.xlim(left=0)
        plt.ylim(bottom=0)

        if self.save:
            plt.savefig(self.path + "Loss.png")

    def main(self, path: str = None, save: bool = True) -> None:
        self.save = save

        self.path = path

        losses = self.load_model_data()

        config = load_config(self.path)

        self.goal_line = config[f"dftb_dft_delta_{config['loss_metric']}"]

        self.plot_loss(losses)


if __name__ == "__main__":
    Plotter().main()
