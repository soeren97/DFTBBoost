import os
from datetime import datetime

import optuna
import torch
import yaml
from optuna import Trial
from torchmetrics import MeanAbsoluteError as MAE  # noqa: F401
from torchmetrics import MeanAbsolutePercentageError as MAPE  # noqa: F401
from torchmetrics import MeanSquaredError as MSE  # noqa: F401
from torchmetrics import MeanSquaredLogError as MSLE  # noqa: F401

from source.models.model_handler import ModelTrainer
from source.models.models import GNN, GNN_MG, GNN_MG_FO, NN  # noqa: F401

# optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model_optimization(model_handler: ModelTrainer, trial: Trial) -> float:
    """Train model for optimization.

    Args:
        model_handler (ModelTrainer): Object used to handle the model.
        trial (Trial): Current trail.

    Raises:
        optuna.TrialPruned: If Trial is pruned stop training and go to next trial.

    Returns:
        float: Loss of trial.
    """
    for epoch in range(model_handler.epochs):
        (
            loss_train,
            pred_train,
            Y_train,
            pred_energy_train,
            Y_energy_train,
        ) = model_handler.train()

        (
            loss_test,
            pred_test,
            Y_test,
            pred_energy_test,
            Y_energy_test,
        ) = model_handler.test()

        model_handler.scheduler.step(loss_test)

        model_handler.evaluate_early_stopping(loss_test)

        trial.report(loss_test, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if model_handler.early_stopping:
            return loss_test

    return loss_test


def hyperparameter_objective(trial: Trial, trainer: ModelTrainer) -> float:
    """Function to be optimized.

    Args:
        trial (Trial): Current trial.
        trainer (ModelTrainer): Object used to handle the model.

    Returns:
        float: Loss of current trial.
    """
    embeding_size = trial.suggest_int("embedding_size", 4, 28)

    trainer.model = GNN(embeding_size).to(trainer.device)
    trainer.loss_metric = "All"

    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.95)
    beta2 = trial.suggest_float("beta2", 0.951, 0.99999)
    decay_rate = trial.suggest_float("decay_rate", 0.001, 0.01)

    scheduler_patience = 5

    trainer.loss_fn = MSE()
    trainer.reset_patience = 20
    trainer.patience = 20
    trainer.best_loss = 10e2
    trainer.early_stopping = False

    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=decay_rate,
    )
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        patience=scheduler_patience,
    )

    test_loss = train_model_optimization(trainer, trial)

    del trainer.model

    return test_loss


def optimize_model():
    """Optimize model."""
    now = datetime.now().strftime("%y_%m_%d_%H_%M")

    model_trainer = ModelTrainer()
    model_trainer.epochs = 300
    model_trainer.data_intervals = os.listdir("Data/datasets")
    model_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer.model = GNN(2).to(model_trainer.device)
    model_trainer.batch_size = int(2048 / 32)
    model_trainer.setup_data()

    model_name = model_trainer.model.__class__.__name__

    pruner = optuna.pruners.SuccessiveHalvingPruner()

    folder = f"Optuna/{now}/"

    os.makedirs(folder)

    storage = f"sqlite:///{folder}study.db"

    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name=model_name, storage=storage
    )

    study.optimize(
        lambda trail: hyperparameter_objective(trail, model_trainer),
        n_trials=100,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Get the best trial from the study
    best_trial = study.best_trial

    # Convert the trial to a dictionary
    trial_dict = best_trial.params

    # Save the dictionary to a YAML file
    with open(f"{folder}{model_name}.yaml", "w+") as outfile:
        yaml.dump(trial_dict, outfile, default_flow_style=False)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


if __name__ == "__main__":
    optimize_model()
