import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from plotly.graph_objects import Figure

import optuna
from optuna.visualization import (
    plot_param_importances,
    plot_optimization_history,
    plot_parallel_coordinate,
)


def load_study(path: Optional[str] = None) -> Tuple[optuna.study.Study, str]:
    root_dir = "Optuna/"
    if path == None:
        studies = os.listdir(root_dir)

        studies = sorted(
            studies, key=lambda x: os.path.getmtime(os.path.join(root_dir, x))
        )
        folder = os.path.join(root_dir, studies[-1], "study.db")
    else:
        folder = os.path.join(root_dir, path, "study.db")

    storage = f"sqlite:///{folder}"

    study = optuna.load_study(storage=storage, study_name="GNN")

    return study, folder[:-8]


def save_image(
    path: str, plotly_figure: Figure, update_traces: Optional[bool] = False
) -> None:
    plotly_figure.update_layout(plot_bgcolor="white")

    if update_traces:
        plotly_figure.update_traces(texttemplate=None, textposition="outside", text="")

    fig.update_layout(title="")

    fig.update_layout(autosize=True)

    plotly_figure.write_image(os.path.join(save_folder, path))


if __name__ == "__main__":
    # Load study
    completed_study, save_folder = load_study("23_04_13_15_25")

    # Plot importance
    fig = plot_param_importances(completed_study)

    save_image("param_imporatnce.png", fig, update_traces=True)

    # Plot history
    fig = plot_optimization_history(completed_study)

    save_image("optimization_history.png", fig)

    # Plot param dependencies
    fig = plot_parallel_coordinate(study=completed_study)

    save_image("param_dependencies.png", fig)
