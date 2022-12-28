import optuna
import os
import torch
import yaml
from datetime import date

from model_handler import ModelTrainer
from models import GNN, CNN, NN, GNN_plus

from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

optuna.logging.set_verbosity(optuna.logging.WARNING)

def hyperparameter_objective(trail, trainer):
    trainer.model = GNN_plus().to(trainer.device)    

    trainer.lr = trail.suggest_float('Learning_rate', 1e-6, 1e-2, log=True)
    trainer.decay_rate = trail.suggest_float('Decay_rate', 1e-4, 0.1, step = 1e-4)
    trainer.batch_size = 2**trail.suggest_int('Batch_size', 8, 10)
    epsilon = trail.suggest_float('Epsilon', 1e-7, 1e-4)
    trainer.loss_fn = MSE()
    trainer.reset_patience = 5

    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), 
                                    lr = trainer.lr, 
                                    eps = epsilon,
                                    weight_decay=trainer.decay_rate
                                    )

    loss = trainer.train_model()

    del trainer.model

    return loss.iloc[0,-1]

model_trainer = ModelTrainer()
model_trainer.epochs = 300
model_trainer.data_intervals = os.listdir('Data/datasets')
model_trainer.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_trainer.model = GNN_plus().to(model_trainer.device)
model_trainer.setup_data()

#pruner = optuna.pruners.SuccessiveHalvingPruner()

study = optuna.create_study(direction = 'minimize', 
                            #pruner=pruner
                            )

study.optimize(lambda trail: hyperparameter_objective(trail, model_trainer), 
               n_trials=50, 
               gc_after_trial = True
               )


# Get the best trial from the study
best_trial = study.best_trial

# Convert the trial to a dictionary
trial_dict = best_trial.params

# Save the dictionary to a YAML file
with open(f'Optuna/{date.today().strftime("%y_%m_%d")}.yaml', 'w+') as outfile:
    yaml.dump(trial_dict, outfile, default_flow_style=False)