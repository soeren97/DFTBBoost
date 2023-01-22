import torch
from torch.nn import Linear, Dropout2d, Conv2d, Module
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GATConv, BatchNorm


class GNN(Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.embedding_size = 16

        self.initial_conv = GATConv(4, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)
        self.batchnorm = BatchNorm(self.embedding_size)
        self.out = Linear(self.embedding_size * 2, 2145 * 2)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch
        edge_attr = data.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.batchnorm(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        out = self.out(hidden)

        return out


class GNN_plus(Module):
    def __init__(self):
        super(GNN_plus, self).__init__()
        self.embedding_size = 64

        self.initial_conv = GATConv(8, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)

        self.batchnorm = BatchNorm(self.embedding_size)

        self.out = Linear(self.embedding_size * 2, 2145 * 2)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch
        edge_attr = data.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.batchnorm(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        out = self.out(hidden)

        return out


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = Conv2d(
            in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1
        )
        self.dropout = Dropout2d(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class NN(Module):
    def __init__(self):
        super(NN, self).__init__()
        self.embedding_size = 64

        self.initial = Linear(2145 * 2, self.embedding_size)
        self.lin1 = Linear(self.embedding_size, self.embedding_size * 2)
        self.lin2 = Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.out = Linear(self.embedding_size * 4, 2145 * 2)

    def forward(self, x):
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


models = [NN(), GNN(), GNN_plus()]

if __name__ == "__main__":
    for model in models:
        model_name = model.__class__.__name__
        print(
            f"{model_name} has {sum(p.numel() for p in model.parameters())} parameters"
        )
