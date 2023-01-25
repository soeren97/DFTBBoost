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

from typing import List


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
        self.decay_rate = None
        self.loss_metric = None

        self.epochs = None
        self.batch_size = None

        self.train_loader = None
        self.test_loader = None
        self.data_loader = None
        self.collate_fn = None

        self.optimizer = None

        self.data = None
        self.train_set = None
        self.test_set = None
        self.valid_set = None

        self.model_folder = None
        self.save_model = None

    def setup_data(self) -> None:
        model_name = self.model.__class__.__name__

        if model_name in ["GNN", "GNN_plus"]:
            self.loader = GNNDataloader
            self.collate_fn = utils.costume_collate_GNN
        else:
            self.loader = torch.utils.data.DataLoader
            self.collate_fn = utils.costume_collate_NN

        self.data = CostumDataset(ml_method=model_name)

        self.train_set, self.test_set = utils.random_split(self.data, [0.8, 0.2])

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

    def train(self) -> torch.Tensor:  # test new data
        self.model.train()
        for batch in self.train_loader:

            self.optimizer.zero_grad()

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

            if self.model.__class__.__name__ in ["GNN", "GNN_plus"]:
                preds = self.model(X)

            else:
                preds = self.model(X.float())

            loss = self.evaluate_loss(preds, n_electrons, n_orbitals, Y)

            loss.backward()

            # Update using the gradients
            self.optimizer.step()

        return loss

    def test(self) -> torch.Tensor:
        self.model.eval()
        for batch in self.test_loader:
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
            if self.model.__class__.__name__ in ["GNN", "GNN_plus"]:
                preds = self.model(X)

            else:
                preds = self.model(X.float())

        loss = self.evaluate_loss(preds, n_electrons, n_orbitals, Y)

        return loss

    def train_model(self) -> pd.DataFrame:
        losses_train = []
        losses_test = []

        self.train_loader = self.loader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = self.loader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in (
            pbar := tqdm(
                range(self.epochs), total=self.epochs, desc="Training", leave=False
            )
        ):
            loss_train = self.train()
            losses_train.append(loss_train.cpu().detach())

            loss_test = self.test()
            losses_test.append(loss_test.cpu().detach())

            pbar.set_description(f"Test loss {loss_test:.2E}")

            self.evaluate_early_stopping(loss_test)

            if self.early_stopping and self.save_model:
                print(f" Best loss {self.best_loss}")
                print(self.model_folder)

                losses = np.array([losses_train, losses_test]).T

                return pd.DataFrame(losses, columns=["Train_loss", "Test_loss"])

            elif self.early_stopping:
                losses = np.array([losses_train, losses_test]).T

                return pd.DataFrame(losses, columns=["Train_loss", "Test_loss"])

        losses = np.array([losses_train, losses_test]).T

        return pd.DataFrame(losses, columns=["Train_loss", "Test_loss"])

    def main(self) -> None:
        self.model_folder = "Models/m" + str(time.time())[:-8] + "/"
        os.mkdir(self.model_folder)
        shutil.copy("model_config/config.yaml", self.model_folder + "config.yaml")

        config = utils.load_config()
        self.epochs = config["epochs"]
        self.batch_size = int(config["batch_size"] / 32)
        self.decay_rate = float(config["decay_rate"])
        self.lr = float(config["lr"])
        self.reset_patience = config["start_patience"]
        self.model = eval(config["model"])().to(self.device)
        self.loss_metric = config["loss_metric"]

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.decay_rate
        )

        self.loss_fn = eval(config["loss_fn"])().to(self.device)

        self.data_intervals = os.listdir("Data/datasets")

        self.setup_data()

        loss_df = self.train_model()

        torch.save(self.model, self.model_folder + "model.pkl")
        loss_df.to_pickle(self.model_folder + "losses.pkl")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.main()
