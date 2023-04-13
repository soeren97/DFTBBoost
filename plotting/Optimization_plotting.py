import os
from typing import Optional, Tuple

import optuna
from optuna.visualization import (
    plot_param_importances,
    plot_optimization_history,
    plot_parallel_coordinate,
)


def load_study(path: Optional[str] = None) -> Tuple[optuna.study.Study, str]:
    root_dir = "Optuna/"
    if path == None:
        studies = os.listdir("root_dir")

        studies = sorted(
            studies, key=lambda x: os.path.getmtime(os.path.join("Optuna", x))
        )
        folder = os.path.join(root_dir, studies[-1])
    else:
        folder = os.path.join(root_dir, path)

    study = optuna.load_study(storage=f"{folder}study.db")

    return study, folder


if __name__ == "__main__":
    # Load study
    completed_study, save_folder = load_study()

    # Plot importance
    fig = plot_param_importances(completed_study)
    fig.write_image(os.path.join(save_folder, "param_imporatnce.png"))

    # Plot history
    fig = plot_optimization_history(completed_study)
    fig.write_image(os.path.join(save_folder, "optimization_history.png"))

    # Plot param dependencies
    fig = plot_parallel_coordinate(study=completed_study)
    fig.write_image(os.path.join(save_folder, "param_dependencies.png"))
