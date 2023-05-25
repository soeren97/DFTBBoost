"""Script used to control models."""
import os
import shutil
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from models import GNN, GNN_MG, GNN_MG_FO, NN  # noqa: F401
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GNNDataloader
from torchmetrics import MeanAbsoluteError as MAE  # noqa: F401
from torchmetrics import MeanAbsolutePercentageError as MAPE  # noqa: F401
from torchmetrics import MeanSquaredError as MSE  # noqa: F401
from torchmetrics import MeanSquaredLogError as MSLE  # noqa: F401
from tqdm import tqdm

import source.utils as utils
from source.data.CostumDataset import CostumDataset


class ModelTrainer:
    """Class used to train the models."""

    def __init__(self) -> None:
        """Initialize class."""
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
        self.model_name: str

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
        """Initialize data and data loader."""
        self.model_name = self.model.__class__.__name__

        if self.model_name in ["GNN", "GNN_MG", "GNN_MG_FO"]:
            self.collate_fn = utils.costume_collate_GNN
            self.loader = GNNDataloader
        else:
            self.collate_fn = utils.costume_collate_NN
            self.loader = lambda dataset, batch_size, shuffle: DataLoader(
                dataset, batch_size, shuffle, collate_fn=self.collate_fn
            )

        self.data = CostumDataset(ml_method=self.model_name)

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
            test_set, batch_size=self.batch_size, shuffle=False
        )

        self.valid_loader = self.loader(
            valid_set, batch_size=self.batch_size, shuffle=False
        )

    def evaluate_early_stopping(self, loss: Tensor) -> None:
        """Evaluate early stopping based on loss.

        Args:
            loss (Tensor): Current loss of a model.
        """
        if not self.best_loss >= loss:
            self.patience -= 1
        else:
            self.best_loss = loss
            self.patience = self.reset_patience

        if self.patience == 0:
            self.early_stopping = True

    def evaluate_loss(
        self,
        preds: Tensor,
        n_electrons: Tensor,
        n_orbitals: Tensor,
        true: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate current loss.

        Args:
            preds (Tensor): Predicted values.
            n_electrons (Tensor): Number of electrons in the molecules.
            n_orbitals (Tensor): Number of orbitals in the molecules.
            true (List[Tensor]): True values.

        Returns:
            Tuple[Tensor,
            Tensor,
            Tensor
            ]: Loss of the batch, predicted values, true values.
        """
        # Calculate eigenvalues
        eigenvalues = utils.find_eigenvalues(preds, n_electrons, n_orbitals)

        if self.loss_metric == "All":
            predicted = eigenvalues
            truth = true[0]

        else:
            # TODO: Does not work,
            # take from find_eigenvalues
            # as it already zero out unwanted orbitals
            predicted = utils.zero_prediction_padding(preds, n_orbitals)
            truth = true[1]

        loss = self.loss_fn(predicted, truth)

        return loss, predicted, truth

    def closure(
        self, batch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,]:
        """Calculate loss for a batch.

        Args:
            batch (_type_): Batch of molecules.

        Returns:
            Tuple[ Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            ]: List of properties and predictions.
        """
        if self.model_name != "NN":
            (
                X,
                Y_eigenvalues,
                Y_matrices,
                n_electrons,
                n_orbitals,
            ) = self.collate_fn(batch)
        else:
            (
                X,
                Y_eigenvalues,
                Y_matrices,
                n_electrons,
                n_orbitals,
            ) = batch

        Y = [Y_eigenvalues, Y_matrices]

        preds = self.model(X)

        loss, pred_energy, true_energy = self.evaluate_loss(
            preds, n_electrons, n_orbitals, Y
        )

        return loss, preds, Y_matrices, n_orbitals, pred_energy, true_energy

    def train(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
        """Train on a batch.

        Returns:
            Tuple[Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            ]: List of properties and predictions.
        """
        self.optimizer.zero_grad()
        self.model.train()
        for batch in self.train_loader:
            (
                loss,
                preds,
                Y_matrices,
                n_orbitals,
                pred_energy,
                true_energy,
            ) = self.closure(batch)

            loss.backward()
            self.optimizer.step()

        return loss, preds, Y_matrices, pred_energy, true_energy

    def test(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
        """Test on a batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
            List of properties and predictions.
        """
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
        """Validate on a batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,]:
            List of properties and predictions.
        """
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
    ) -> pd.DataFrame:
        """Train a model on the entrie dataset

        Returns:
            pd.DataFrame: Train, test and validation loss.
        """
        losses_train = []

        losses_test = []

        losses_valid = []

        for epoch in (
            pbar := tqdm(
                range(self.epochs),
                total=self.epochs,
                desc="Training",
                leave=False,
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
                break

        losses = self.save_prediction(
            losses_train,
            losses_test,
            losses_valid,
            Y_train,
            pred_train,
            pred_energy_train,
            Y_energy_train,
            Y_test,
            pred_test,
            pred_energy_test,
            Y_energy_test,
            Y_valid,
            pred_valid,
            pred_energy_valid,
            Y_energy_valid,
        )

        return pd.DataFrame(losses, columns=["Train_loss", "Test_loss", "Valid_loss"])

    def save_prediction(
        self,
        losses_train: List,
        losses_test: List,
        losses_valid: List,
        Y_train: Tensor,
        pred_train: Tensor,
        pred_energy_train: Tensor,
        Y_energy_train: Tensor,
        Y_test: Tensor,
        pred_test: Tensor,
        pred_energy_test: Tensor,
        Y_energy_test: Tensor,
        Y_valid: Tensor,
        pred_valid: Tensor,
        pred_energy_valid: Tensor,
        Y_energy_valid: Tensor,
    ) -> NDArray:
        """Save the predictions, loss and energies of the model.

        Args:
            losses_train (List): Training loss.
            losses_test (List): Test loss.
            losses_valid (List): Validation loss.
            Y_train (Tensor): Training target.
            pred_train (Tensor): Training prediction.
            pred_energy_train (Tensor): Training energy prediction.
            Y_energy_train (Tensor): Training energy target.
            Y_test (Tensor): Test target.
            pred_test (Tensor): Test prediction.
            pred_energy_test (Tensor): Test energy prediction.
            Y_energy_test (Tensor): Test energy target.
            Y_valid (Tensor): Validation target.
            pred_valid (Tensor): Validation prediction.
            pred_energy_valid (Tensor): Validation energy prediction
            Y_energy_valid (Tensor): Validation energy target.

        Returns:
            NDArray: Training, test and validation loss.
        """
        losses = np.array([losses_train, losses_test, losses_valid]).T
        if self.save:
            train_df = pd.DataFrame(
                {
                    "pred": [pred_train.cpu().detach().numpy()],
                    "y": [Y_train.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_train.cpu().detach().numpy()],
                    "energy_true": [Y_energy_train.cpu().detach().numpy()],
                }
            )

            train_df.to_pickle(self.folder + "predictions/train.pkl")

            test_df = pd.DataFrame(
                {
                    "pred": [pred_test.cpu().detach().numpy()],
                    "y": [Y_test.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_test.cpu().detach().numpy()],
                    "energy_true": [Y_energy_test.cpu().detach().numpy()],
                }
            )

            test_df.to_pickle(self.folder + "predictions/test.pkl")

            valid_df = pd.DataFrame(
                {
                    "pred": [pred_valid.cpu().detach().numpy()],
                    "y": [Y_valid.cpu().detach().numpy()],
                    "energy_pred": [pred_energy_valid.cpu().detach().numpy()],
                    "energy_true": [Y_energy_valid.cpu().detach().numpy()],
                }
            )

            valid_df.to_pickle(self.folder + "predictions/valid.pkl")
        return losses

    def main(self) -> None:
        """Run the entire script."""
        model_folder = "Models/m" + str(time.time())[:-8] + "/"

        print(model_folder)

        self.save = True

        if self.save:
            os.mkdir(model_folder)
            os.mkdir(model_folder + "predictions/")
            shutil.copy("model_config/config.yaml", model_folder + "config.yaml")

        config = utils.load_config(model_name="GNN")
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
