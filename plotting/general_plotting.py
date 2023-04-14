import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_tanh():
    save_location = os.path.join(os.getcwd(), "Figures", "tanh.png")

    x = np.linspace(-5, 5, 100)

    y = torch.tanh(torch.tensor(x)).numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y, zorder=-1)
    ax.set_xlabel("x")
    ax.set_ylabel("tanh(x)")

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    yticks = [-1, -0.5, 0.5, 1]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.xaxis.set_ticks_position("bottom")

    ax.xaxis.set_label_coords(0.5, -0.02)
    ax.yaxis.set_label_coords(-0.02, 0.5)

    plt.tight_layout()
    plt.savefig(save_location, dpi=300)


if __name__ == "__main__":
    plot_tanh()
