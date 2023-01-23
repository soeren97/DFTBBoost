import optuna
import os
import torch
import yaml
from datetime import datetime

from model_handler import ModelTrainer
from models import GNN, NN, GNN_plus

from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

# optuna.logging.set_verbosity(optuna.logging.WARNING)


def hyperparameter_objective(trail: optuna.Trial, trainer: ModelTrainer) -> float:
    trainer.model = GNN_plus().to(trainer.device)
    trainer.loss_metric = "All"

    trainer.lr = trail.suggest_float("Learning_rate", 1e-9, 1e-5, log=True)
    trainer.decay_rate = trail.suggest_float("Decay_rate", 1e-4, 0.1, step=1e-4)
    trainer.batch_size = 2 ** trail.suggest_int("Batch_size", 9, 12)
    epsilon = trail.suggest_float("Epsilon", 1e-7, 1e-4)
    trainer.loss_fn = MSE()
    trainer.reset_patience = 20
    trainer.patience = 50
    trainer.early_stopping = False

    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(),
        lr=trainer.lr,
        eps=epsilon,
        weight_decay=trainer.decay_rate,
    )

    loss = trainer.train_model()

    del trainer.model

    return loss.iloc[0, -1]


def optimize_model():
    now = datetime.now().strftime("%y_%m_%d_%H%M%S")

    model_trainer = ModelTrainer()
    model_trainer.epochs = 1000
    model_trainer.data_intervals = os.listdir("Data/datasets")
    model_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer.model = GNN().to(model_trainer.device)
    model_trainer.setup_data()

    model_name = (model_trainer.model.__class__.__name__,)

    pruner = optuna.pruners.SuccessiveHalvingPruner()

    # storage = optuna.storages.RDBStorage(f'/Optuna/studies/{now}.db')
    # storage = f"sqlite:///Optuna/studies/{now}.db"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name=model_name
    )

    study.optimize(
        lambda trail: hyperparameter_objective(trail, model_trainer),
        n_trials=50,
        gc_after_trial=True,
    )

    # Get the best trial from the study
    best_trial = study.best_trial

    # Convert the trial to a dictionary
    trial_dict = best_trial.params

    # Save the dictionary to a YAML file
    with open(f"Optuna/{model_name + now}.yaml", "w+") as outfile:
        yaml.dump(trial_dict, outfile, default_flow_style=False)


if __name__ == "__main__":
    optimize_model()
