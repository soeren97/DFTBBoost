"""Models used in the project."""
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import BatchNorm, GATConv, GraphNorm
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GNN(Module):
    """Graph Convolutional Network class."""

    def __init__(self, embedding_size: int) -> None:
        """Initialize network.

        Args:
            embedding_size (int): Embedding size, controls the size of the network.
        """
        super(GNN, self).__init__()
        self.embedding_size = embedding_size

        self.initial_conv = GATConv(1, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)
        self.conv2 = GATConv(self.embedding_size, self.embedding_size)
        self.batchnorm = BatchNorm(self.embedding_size)
        self.graphnorm = GraphNorm(self.embedding_size)
        self.out = Linear(self.embedding_size * 2, 2145 * 2)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the network on data.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output.
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch
        edge_attr = data.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.batchnorm(hidden)
        hidden = self.graphnorm(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # hidden = self.batchnorm(hidden)
        # hidden = self.graphnorm(hidden)

        # hidden = self.conv1(hidden, edge_index, edge_attr)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        out = self.out(hidden)

        return out


class GNN_MG(Module):
    """Graph Neural Network containing Molecular Geometry"""

    def __init__(self, embedding_size):
        """Initialize network.

        Args:
            embedding_size (int): Embedding size, controls the size of the network.
        """
        super(GNN_MG, self).__init__()
        self.embedding_size = embedding_size

        self.initial_conv = GATConv(4, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)
        self.batchnorm = BatchNorm(self.embedding_size)
        self.graphnorm = GraphNorm(self.embedding_size)
        self.out = Linear(self.embedding_size * 2, 2145 * 2)

    def forward(self, data):
        """Apply the network on data.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output.
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch
        edge_attr = data.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.batchnorm(hidden)
        hidden = self.graphnorm(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        out = self.out(hidden)

        return out


class GNN_MG_FO(Module):
    """
    Graph Neural Network containing Molecular Geometry, Fock matrix and Overlap matrix.
    """

    def __init__(self, embedding_size):
        """Initialize network.

        Args:
            embedding_size (int): Embedding size, controls the size of the network.
        """
        super(GNN_MG_FO, self).__init__()
        self.embedding_size = embedding_size

        self.initial_conv = GATConv(8, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)

        self.batchnorm = BatchNorm(self.embedding_size)
        self.graphnorm = GraphNorm(self.embedding_size)

        self.out = Linear(self.embedding_size * 2, 2145 * 2)

    def forward(self, data):
        """Apply the network on data.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output.
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch
        edge_attr = data.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.batchnorm(hidden)
        hidden = self.graphnorm(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        out = self.out(hidden)

        return out


class NN(Module):
    """Simple fully connected neural network."""

    def __init__(self, embedding_size):
        """Initialize network.

        Args:
            embedding_size (int): Embedding size, controls the size of the network.
        """
        super(NN, self).__init__()
        self.embedding_size = embedding_size

        self.initial = Linear(2145 * 2, self.embedding_size)
        self.lin1 = Linear(self.embedding_size, self.embedding_size * 2)
        self.lin2 = Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.out = Linear(self.embedding_size * 4, 2145 * 2)

    def forward(self, x):
        """Apply the network on data.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output.
        """
        hidden = self.initial(x)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.lin1(hidden)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.lin2(hidden)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        out = self.out(hidden)

        return out


models = [NN(16), GNN(16), GNN_MG(16), GNN_MG_FO(16)]

if __name__ == "__main__":
    for model in models:
        model_name = model.__class__.__name__
        print(
            f"{model_name} has {sum(p.numel() for p in model.parameters())} parameters"
        )
