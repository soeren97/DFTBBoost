from tqdm import tqdm
import os
import time
import shutil
import utils

import pandas as pd
import numpy as np
import torch

from torch_geometric.loader import DataLoader as GNNDataloader
from torch_geometric.data import Batch

from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

from models import GNN, CNN, NN, GNN_plus, GNN_minus
from CostumDataset import CostumDataset

from typing import List, Tuple


class ModelTrainer:
    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.dataloader = None

        self.loss_fn = None
        self.best_loss = 10e2
        self.patience = 50
        self.reset_patience = None
        self.early_stopping = False
        self.lr = None
        self.loss_metric = None

        self.epochs = None
        self.batch_size = None

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.data_loader = None
        self.collate_fn = None

        self.optimizer = None
        self.scheduler = None

        self.data = None
        self.train_set = None
        self.test_set = None
        self.valid_set = None

    def setup_data(self) -> None:
        model_name = self.model.__class__.__name__

        if model_name in ["GNN", "GNN_plus", "GNN_minus"]:
            self.loader = GNNDataloader
            self.collate_fn = utils.costume_collate_GNN
        else:
            self.loader = torch.utils.data.DataLoader
            self.collate_fn = utils.costume_collate_NN

        self.data = CostumDataset(ml_method=model_name)

        self.train_set, self.test_set, self.valid_set = utils.random_split(
            self.data, [0.8, 0.1, 0.1]
        )

    def evaluate_early_stopping(self, loss: torch.Tensor) -> None:
        if not self.best_loss >= loss:
            self.patience -= 1
        else:
            self.best_loss = loss
            self.patience = self.reset_patience

        if self.patience == 0:
            self.early_stopping = True

    def evaluate_loss(
        self,
        preds: torch.Tensor,
        n_electrons: torch.Tensor,
        n_orbitals: torch.Tensor,
        true: List[torch.Tensor],
    ) -> torch.Tensor:

        # Calculate eigenvalues
        eigenvalues, HOMO, LUMO = utils.find_eigenvalues(preds, n_electrons, n_orbitals)

        if self.loss_metric == "HOMO_LUMO":
            predicted = torch.stack([HOMO, LUMO])
            true = torch.stack(true[:2])

        elif self.loss_metric == "All":
            predicted = eigenvalues
            true = true[2]

        else:
            # TODO: Does not work, take from find_eigenvalues as it already zero out unwanted orbitals
            predicted = utils.zero_prediction_padding(preds, n_orbitals)
            true = true[3]

        loss = self.loss_fn(predicted, true)

        return loss

    def closure(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            X,
            Y_HOMO,
            Y_LUMO,
            Y_eigenvalues,
            Y_matrices,
            n_electrons,
            n_orbitals,
        ) = self.collate_fn(batch)

        Y = [Y_HOMO, Y_LUMO, Y_eigenvalues, Y_matrices]

        n_electrons.to(self.device)

        n_orbitals.to(self.device)

        if self.model.__class__.__name__ in ["GNN", "GNN_plus", "GNN_minus"]:
            preds = self.model(X)

        else:
            preds = self.model(X.float())

        loss = self.evaluate_loss(preds, n_electrons, n_orbitals, Y)

        loss.backward()

        return loss, preds, Y_matrices, n_orbitals

    def train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        self.model.train()
        for batch in self.train_loader:
            # loss = self.optimizer.step(lambda: self.closure(batch))
            loss, preds, Y_matrices, n_orbitals = self.closure(batch)
            self.optimizer.step()

        return loss, preds, Y_matrices, n_orbitals

    def test(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        for batch in self.test_loader:
            loss, preds, Y_matrices, n_orbitals = self.closure(batch)

        return loss, preds, Y_matrices, n_orbitals

    def validate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        for batch in self.valid_loader:
            loss, preds, Y_matrices, n_orbitals = self.closure(batch)

        return loss, preds, Y_matrices, n_orbitals

    def train_model(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        losses_train = []
        preds_train = []
        ys_train = []
        n_orbitals_train = []

        losses_test = []
        preds_test = []
        ys_test = []
        n_orbitals_test = []

        losses_valid = []
        preds_valid = []
        ys_valid = []
        n_orbitals_valid = []

        self.train_loader = self.loader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = self.loader(
            self.test_set,
            batch_size=self.batch_size,
        )

        self.valid_loader = self.loader(
            self.valid_set,
            batch_size=self.batch_size,
        )

        for epoch in (
            pbar := tqdm(
                range(self.epochs), total=self.epochs, desc="Training", leave=False
            )
        ):
            loss_train, pred_train, Y_train, n_orbitals = self.train()
            losses_train.append(loss_train.cpu().detach())
            preds_train.append(pred_train.cpu().detach().numpy())
            ys_train.append(Y_train.cpu().detach().numpy())
            n_orbitals_train.extend(n_orbitals.cpu().detach())

            loss_test, pred_test, Y_test, n_orbitals = self.test()
            losses_test.append(loss_test.cpu().detach())
            preds_test.append(pred_test.cpu().detach().numpy())
            ys_test.append(Y_test.cpu().detach().numpy())
            n_orbitals_test.extend(n_orbitals.cpu().detach())

            self.scheduler.step(loss_test)

            loss_valid, pred_valid, Y_valid, n_orbitals = self.validate()
            losses_valid.append(loss_valid.cpu().detach())
            preds_valid.append(pred_valid.cpu().detach().numpy())
            ys_valid.append(Y_valid.cpu().detach().numpy())
            n_orbitals_valid.extend(n_orbitals.cpu().detach())

            pbar.set_description(f"Test loss {loss_test:.2E}")

            self.evaluate_early_stopping(loss_test)

            if self.early_stopping:
                losses = np.array([losses_train, losses_test, losses_valid]).T

                return (
                    pd.DataFrame(
                        losses, columns=["Train_loss", "Test_loss", "Valid_loss"]
                    ),
                    pd.DataFrame(
                        {
                            "pred": np.vstack(preds_train).tolist(),
                            "y": np.vstack(ys_train).tolist(),
                            "n_orbitals": n_orbitals_train,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "pred": np.vstack(preds_test).tolist(),
                            "y": np.vstack(ys_test).tolist(),
                            "n_orbitals": n_orbitals_test,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "pred": np.vstack(preds_valid).tolist(),
                            "y": np.vstack(ys_valid).tolist(),
                            "n_orbitals": n_orbitals_valid,
                        }
                    ),
                )

        losses = np.array([losses_train, losses_test, losses_valid]).T

        return (
            pd.DataFrame(losses, columns=["Train_loss", "Test_loss", "Valid_loss"]),
            pd.DataFrame(
                {
                    "pred": np.vstack(preds_train).tolist(),
                    "y": np.vstack(ys_train).tolist(),
                }
            ),
            pd.DataFrame(
                {
                    "pred": np.vstack(preds_test).tolist(),
                    "y": np.vstack(ys_test).tolist(),
                }
            ),
            pd.DataFrame(
                {
                    "pred": np.vstack(preds_valid).tolist(),
                    "y": np.vstack(ys_valid).tolist(),
                }
            ),
        )

    def main(self) -> None:
        model_folder = "Models/m" + str(time.time())[:-8] + "/"

        os.mkdir(model_folder)
        os.mkdir(model_folder + "predictions/")
        shutil.copy("model_config/config.yaml", model_folder + "config.yaml")

        config = utils.load_config()
        self.epochs = config["epochs"]
        self.batch_size = int(config["batch_size"] / 32)
        self.decay_rate = float(config["decay_rate"])
        self.lr = float(config["lr"])
        self.reset_patience = config["start_patience"]
        self.model = eval(config["model"])().to(self.device)
        self.loss_metric = config["loss_metric"]

        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.decay_rate
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
        )

        self.loss_fn = eval(config["loss_fn"])().to(self.device)

        self.data_intervals = os.listdir("Data/datasets")

        self.setup_data()

        loss_df, pred_train, pred_test, pred_valid = self.train_model()

        torch.save(self.model, model_folder + "model.pkl")
        loss_df.to_pickle(model_folder + "losses.pkl")
        pred_train.to_pickle(model_folder + "predictions/train.pkl")
        pred_test.to_pickle(model_folder + "predictions/test.pkl")
        pred_valid.to_pickle(model_folder + "predictions/valid.pkl")
        print(model_folder)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.main()
