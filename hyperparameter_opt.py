import optuna
import os
import torch
import yaml
from datetime import datetime

from model_handler import ModelTrainer
from models import GNN_MG, NN, GNN_MG_FO, GNN

from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

# optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model_optimization(model_handler, trial) -> float:
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


def hyperparameter_objective(trial: optuna.Trial, trainer: ModelTrainer) -> float:
    trainer.model = GNN().to(trainer.device)
    trainer.loss_metric = "All"

    lr = trial.suggest_float("Learning_rate", 1e-9, 1e-5, log=True)
    beta1 = trial.suggest_float("Beta1", 0.8, 0.95)
    beta2 = trial.suggest_float("Beta2", 0.951, 0.99999)
    decay_rate = trial.suggest_float("Decay rate", 0.001, 0.01)

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
    now = datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    model_trainer = ModelTrainer()
    model_trainer.epochs = 300
    model_trainer.data_intervals = os.listdir("Data/datasets")
    model_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer.model = GNN().to(model_trainer.device)
    model_trainer.batch_size = int(2048 / 32)
    model_trainer.setup_data()

    model_name = model_trainer.model.__class__.__name__

    pruner = optuna.pruners.SuccessiveHalvingPruner(bootstrap_count=30)

    # storage = optuna.storages.RDBStorage(f'/Optuna/studies/{now}.db')
    storage = f"sqlite:///Optuna/studies/{now}.db"

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
    with open(f"Optuna/{model_name + now}.yaml", "w+") as outfile:
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
