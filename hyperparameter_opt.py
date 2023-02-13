import optuna
import os
import torch
import yaml
from datetime import datetime

from model_handler import ModelTrainer
from models import GNN, NN, GNN_plus, GNN_minus

from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

# optuna.logging.set_verbosity(optuna.logging.WARNING)


def hyperparameter_objective(trial: optuna.Trial, trainer: ModelTrainer) -> float:
    trainer.model = GNN_minus().to(trainer.device)
    trainer.loss_metric = "All"

    trainer.lr = trial.suggest_float("Learning_rate", 1e-9, 1e-5, log=True)
    trainer.batch_size = 2 ** trial.suggest_int("Batch_size", 9, 12)
    gamma = trial.suggest_float("Gamma", 1e-3, 0.5)
    step_size = trial.suggest_int("Step size", 50, 150)
    trainer.loss_fn = MSE()
    trainer.reset_patience = 20
    trainer.patience = 50
    trainer.early_stopping = False

    trainer.optimizer = torch.optim.LBFGS(trainer.model.parameters(), lr=trainer.lr)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=100, gamma=0.1
    )

    loss = trainer.train_model()

    del trainer.model

    return loss.iloc[0, -1]


def optimize_model():
    now = datetime.now().strftime("%y_%m_%d_%H%M%S")

    model_trainer = ModelTrainer()
    model_trainer.epochs = 300
    model_trainer.data_intervals = os.listdir("Data/datasets")
    model_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer.model = GNN_minus().to(model_trainer.device)
    model_trainer.setup_data()

    model_name = model_trainer.model.__class__.__name__

    pruner = optuna.pruners.SuccessiveHalvingPruner()

    # storage = optuna.storages.RDBStorage(f'/Optuna/studies/{now}.db')
    # storage = f"sqlite:///Optuna/studies/{now}.db"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name=model_name
    )

    study.optimize(
        lambda trail: hyperparameter_objective(trail, model_trainer),
        n_trials=100,
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
