import os
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

sns.set_theme()


class Plotter:
    def __init__(self) -> None:
        self.path = None
        self.save = False

    def load_model_data(self) -> pd.DataFrame:
        if self.path == None:
            folders = os.listdir("Models/")
            folders = sorted(
                folders, key=lambda x: os.path.getmtime(os.path.join("Models", x))
            )
            self.path = "Models/" + folders[-1] + "/"
        return pd.read_pickle(self.path + "losses.pkl")

    def plot_loss(self, loss: pd.DataFrame) -> None:
        loss_train = loss.Train_loss
        loss_test = loss.Test_loss
        plt.plot(loss_train[1:], label="Train")
        plt.plot(loss_test[1:], label="Test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.tight_layout()
        # plt.ylim(top = 7)
        # plt.yscale('log')
        if self.save:
            plt.savefig(self.path + "Loss.png")

    def main(self, path: str = None, save: bool = True) -> None:
        self.save = save

        self.path = path

        losses = self.load_model_data()

        self.plot_loss(losses)


if __name__ == "__main__":
    Plotter().main()
