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

from models import GNN_MG, NN, GNN_MG_FO, GNN
from CostumDataset import CostumDataset

from typing import List, Tuple
from numpy.typing import NDArray


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

        self.folder = None
        self.save = False

    def setup_data(self) -> None:
        model_name = self.model.__class__.__name__

        if model_name in ["GNN", "GNN_MG", "GNN_MG_FO"]:
            self.loader = GNNDataloader
            self.collate_fn = utils.costume_collate_GNN
        else:
            self.loader = torch.utils.data.DataLoader
            self.collate_fn = utils.costume_collate_NN

        self.data = CostumDataset(ml_method=model_name)

        # Ensures same split each time
        generator = torch.Generator().manual_seed(42)

        train_set, test_set, valid_set = utils.random_split(
            self.data, [0.8, 0.1, 0.1], generator=generator
        )

        self.train_loader = self.loader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = self.loader(
            test_set,
            batch_size=self.batch_size,
        )

        self.valid_loader = self.loader(
            valid_set,
            batch_size=self.batch_size,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Calculate eigenvalues
        eigenvalues = utils.find_eigenvalues(preds, n_electrons, n_orbitals)

        if self.loss_metric == "All":
            predicted = eigenvalues
            truth = true[0]

        else:
            # TODO: Does not work, take from find_eigenvalues as it already zero out unwanted orbitals
            predicted = utils.zero_prediction_padding(preds, n_orbitals)
            truth = true[1]

        loss = self.loss_fn(predicted, truth)

        return loss, predicted, truth

    def closure(
        self, batch
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            X,
            Y_eigenvalues,
            Y_matrices,
            n_electrons,
            n_orbitals,
        ) = self.collate_fn(batch)

        Y = [Y_eigenvalues, Y_matrices]

        n_electrons.to(self.device)

        n_orbitals.to(self.device)

        if self.model.__class__.__name__ in ["GNN", "GNN_MG", "GNN_MG_FO"]:
            preds = self.model(X)

        else:
            preds = self.model(X.float())

        loss, pred_energy, true_energy = self.evaluate_loss(
            preds, n_electrons, n_orbitals, Y
        )

        loss.backward()

        return loss, preds, Y_matrices, n_orbitals, pred_energy, true_energy

    def train(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        self.optimizer.zero_grad()
        self.model.train()
        for batch in self.train_loader:
            # loss = self.optimizer.step(lambda: self.closure(batch))
            (
                loss,
                preds,
                Y_matrices,
                n_orbitals,
                pred_energy,
                true_energy,
            ) = self.closure(batch)
            self.optimizer.step()

        return loss, preds, Y_matrices, pred_energy, true_energy

    def test(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        self.model.eval()
        for batch in self.test_loader:
            (
                loss,
                preds,
                Y_matrices,
                n_orbitals,
                pred_energy,
                true_energy,
            ) = self.closure(batch)

        return loss, preds, Y_matrices, pred_energy, true_energy

    def validate(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        self.model.eval()
        for batch in self.valid_loader:
            (
                loss,
                preds,
                Y_matrices,
                n_orbitals,
                pred_energy,
                true_energy,
            ) = self.closure(batch)

        return loss, preds, Y_matrices, pred_energy, true_energy

    def train_model(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        losses_train = []

        losses_test = []

        losses_valid = []

        for epoch in (
            pbar := tqdm(
                range(self.epochs), total=self.epochs, desc="Training", leave=False
            )
        ):
            (
                loss_train,
                pred_train,
                Y_train,
                pred_energy_train,
                Y_energy_train,
            ) = self.train()
            losses_train.append(loss_train.cpu().detach())

            (
                loss_test,
                pred_test,
                Y_test,
                pred_energy_test,
                Y_energy_test,
            ) = self.test()
            losses_test.append(loss_test.cpu().detach())

            self.scheduler.step(loss_test)

            (
                loss_valid,
                pred_valid,
                Y_valid,
                pred_energy_valid,
                Y_energy_valid,
            ) = self.validate()
            losses_valid.append(loss_valid.cpu().detach())

            pbar.set_description(f"Test loss {loss_test:.2E}")

            self.evaluate_early_stopping(loss_test)

            if self.early_stopping:
                losses = self.save_prediction(
                    losses_train,
                    losses_test,
                    losses_valid,
                    Y_train,
                    pred_energy_train,
                    Y_energy_train,
                    Y_test,
                    pred_energy_test,
                    Y_energy_test,
                    Y_valid,
                    pred_energy_valid,
                    Y_energy_valid,
                )

        losses = self.save_prediction(
            losses_train,
            losses_test,
            losses_valid,
            Y_train,
            pred_energy_train,
            Y_energy_train,
            Y_test,
            pred_energy_test,
            Y_energy_test,
            Y_valid,
            pred_energy_valid,
            Y_energy_valid,
        )

        return pd.DataFrame(losses, columns=["Train_loss", "Test_loss", "Valid_loss"])

    def save_prediction(
        self,
        losses_train: List,
        losses_test: List,
        losses_valid: List,
        Y_train: torch.Tensor,
        pred_energy_train: torch.Tensor,
        Y_energy_train: torch.Tensor,
        Y_test: torch.Tensor,
        pred_energy_test: torch.Tensor,
        Y_energy_test: torch.Tensor,
        Y_valid: torch.Tensor,
        pred_energy_valid: torch.Tensor,
        Y_energy_valid: torch.Tensor,
    ) -> NDArray:
        losses = np.array([losses_train, losses_test, losses_valid]).T
        if self.save:
            pred_train = pd.DataFrame(
                {
                    "pred": [pred_train.cpu().detach().numpy()],
                    "y": [Y_train.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_train.cpu().detach().numpy()],
                    "energy_true": [Y_energy_train.cpu().detach().numpy()],
                }
            )

            pred_train.to_pickle(self.folder + "predictions/train.pkl")

            pred_test = pd.DataFrame(
                {
                    "pred": [pred_test.cpu().detach().numpy()],
                    "y": [Y_test.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_test.cpu().detach().numpy()],
                    "energy_true": [Y_energy_test.cpu().detach().numpy()],
                }
            )

            pred_test.to_pickle(self.folder + "predictions/test.pkl")

            pred_valid = pd.DataFrame(
                {
                    "pred": [pred_valid.cpu().detach().numpy()],
                    "y": [Y_valid.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_valid.cpu().detach().numpy()],
                    "energy_true": [Y_energy_valid.cpu().detach().numpy()],
                }
            )

            pred_valid.to_pickle(self.folder + "predictions/valid.pkl")
        return losses

    def main(self) -> None:
        model_folder = "Models/m" + str(time.time())[:-8] + "/"

        print(model_folder)

        self.save = False

        if self.save:
            os.mkdir(model_folder)
            os.mkdir(model_folder + "predictions/")
            shutil.copy("model_config/config.yaml", model_folder + "config.yaml")

        config = utils.load_config()
        embeding_size = config["embedding_size"]
        self.epochs = config["epochs"]
        self.batch_size = int(config["batch_size"] / 32)
        self.decay_rate = float(config["decay_rate"])
        self.lr = float(config["lr"])
        self.reset_patience = config["start_patience"]
        self.model = eval(config["model"])(embeding_size).to(self.device)
        self.loss_metric = config["loss_metric"]

        self.folder = model_folder

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

        loss_df = self.train_model()

        torch.save(self.model, model_folder + "model.pkl")

        loss_df.to_pickle(model_folder + "losses.pkl")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.main()
