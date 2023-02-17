import torch
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from models import GNN_minus, GNN, GNN_plus, NN
import utils
from torch_geometric.loader import DataLoader as GNNDataloader
from CostumDataset import CostumDataset


class Calibrator:
    def __init__(self) -> None:
        self.folder = None
        self.data = None

        self.batch_size = None
        self.model = None
        self.collate_fn = None

        self.test_set = None
        self.train_set = None
        self.valid_set = None

        self.loader = None
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

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

    def setup_loaders(self) -> None:
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

        self.valid_loader = self.loader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def setup_model(self) -> None:
        model = eval(self.config["model"])()
        model = torch.load(f"{self.path}/model.pkl")
        self.model = model

    def isotonic_calibration(self, predictions, targets):
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        calibrated_model = iso_reg.fit(predictions, targets)
        return calibrated_model

    def fit_model(self):
        predictions = []
        targets = []

        for batch in self.train_loader:
            (
                X,
                Y_HOMO,
                Y_LUMO,
                Y_eigenvalues,
                Y_matrices,
                n_electrons,
                n_orbitals,
            ) = self.collate_fn(batch)

            target = Y_eigenvalues

            if self.model.__class__.__name__ in ["GNN", "GNN_plus", "GNN_minus"]:
                output = self.model(X)

            else:
                output = self.model(X.float())

            predictions.append(output.detach().numpy())
            targets.append(target.detach().numpy())

        combined_model = self.isotonic_calibration(predictions, targets)

        return combined_model

    def main(self):
        self.path = "Models/m1676225631/"
        self.config = utils.load_config(self.path)
        self.setup_model()
        self.setup_data()
        self.setup_loaders()
        self.fit_model()


if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.main()
