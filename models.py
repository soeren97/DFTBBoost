import torch
from torch.nn import Linear, Dropout2d, Conv2d  # , conv_transpose2d
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, GATConv, BatchNorm


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.embedding_size = 128

        self.initial_conv = GCNConv(4, self.embedding_size)
        self.conv1 = GCNConv(self.embedding_size, self.embedding_size)
        self.conv2 = GCNConv(self.embedding_size, self.embedding_size)
        self.conv3 = GCNConv(self.embedding_size, self.embedding_size)
        # self.conv4 = GCNConv(self.embedding_size, self.embedding_size)

        self.lin1 = Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.out = Linear(self.embedding_size * 4, 2145 * 2)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch

        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # hidden = self.conv4(hidden, edge_index)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        hidden = self.lin1(hidden)

        out = self.out(hidden)

        return out


class GNN_plus(torch.nn.Module):
    def __init__(self):
        super(GNN_plus, self).__init__()
        self.embedding_size = 8

        self.initial_conv = GATConv(8, self.embedding_size)
        self.conv1 = GATConv(self.embedding_size, self.embedding_size)
        # self.conv2 = GATConv(self.embedding_size, self.embedding_size)
        # self.conv3 = GATConv(self.embedding_size, self.embedding_size)
        # self.conv4 = GATConv(self.embedding_size, self.embedding_size)

        self.batchnorm = BatchNorm(self.embedding_size)

        # self.lin1 = Linear(self.embedding_size, self.embedding_size)
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

        # hidden = self.batchnorm(hidden)

        # hidden = self.conv2(hidden, edge_index, edge_attr)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=.2)

        # hidden = self.batchnorm(hidden)

        # hidden = self.conv3(hidden, edge_index, edge_attr)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=.2)

        # hidden = self.conv4(hidden, edge_index)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=.2)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        # hidden = self.lin1(hidden)

        out = self.out(hidden)

        return out


class CNN(torch.nn.Module):  # fix dimensions
    def __init__(self):
        super(CNN, self).__init__()
        batch_size = 128
        self.embedding_size = 10
        self.kernel_size = 5

        self.dropout = Dropout2d(p=0.2)

        # =============================================================================
        #          self.initial_conv = Conv2d([batch_size,
        #                                     390,
        #                                     195,
        #                                      ],
        #                                     [batch_size,
        #                                      386,
        #                                      191,
        #                                      ],
        #                                     kernel_size = self.kernel_size)
        # =============================================================================

        self.initial_conv = Conv2d(
            batch_size,
            self.embedding_size * 2,
            self.embedding_size,
            kernel_size=self.kernel_size,
        )

        self.conv1 = Conv2d(
            batch_size,
            self.embedding_size,
            self.embedding_size * 2,
            kernel_size=self.kernel_size,
        )

        self.conv2 = Conv2d(
            batch_size,
            self.embedding_size * 2,
            self.embedding_size * 4,
            kernel_size=self.kernel_size,
        )

        self.conv3 = Conv2d(
            batch_size,
            self.embedding_size * 4,
            self.embedding_size * 2,
            kernel_size=self.kernel_size,
        )

        # self.unconv1 = conv_transpose2d()

        # self.conv4 = Conv2d(self.embedding_size, self.embedding_size)

        self.lin1 = Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.out = Linear(self.embedding_size * 4, 2145 * 2)

    def forward(self, x):
        hidden = self.initial_conv(x)  # [256, 386, 191]
        hidden = torch.tanh(hidden)  # [256, 386, 191]
        hidden = self.dropout(hidden)  # [256, 386, 191]

        hidden = self.conv1(hidden)  # [512, 382, 187]
        hidden = torch.tanh(hidden)  # [512, 382, 187]
        hidden = self.dropout(hidden)  # [512, 382, 187]

        hidden = self.conv2(hidden)  # [1024, 378, 183]
        hidden = torch.tanh(hidden)  # [1024, 378, 183]
        hidden = self.dropout(hidden)  # [1024, 378, 183]

        hidden = self.conv3(hidden)  # [512, 374, 179]
        out = torch.tanh(hidden)
        # hidden = Dropout2d(hidden, p=.2)

        # hidden = self.conv4(hidden)
        # hidden = torch.tanh(hidden)
        # hidden = F.dropout(hidden, p=.2)

        # Global Pooling (stack different aggregations)
        # out = torch.cat([gmp(hidden, batch_index),
        #                    gap(hidden, batch_index)], dim=1)

        # hidden = self.lin1(hidden)

        # out = self.out(hidden)

        return out


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.embedding_size = 64

        self.initial = Linear(self.embedding_size, self.embedding_size)
        self.lin1 = Linear(self.embedding_size, self.embedding_size * 2)
        self.lin2 = Linear(self.embedding_size * 2, self.embedding_size * 4)
        self.out = Linear(self.embedding_size * 4, 2145 * 2)

    def forward(self, x):
        hidden = self.initial_conv(x)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.lin1(hidden)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        hidden = self.lin2(hidden)
        hidden = torch.tanh(hidden)
        hidden = F.dropout(hidden, p=0.2)

        # Global Pooling (stack different aggregations)
        # hidden = torch.cat([gmp(hidden, batch_index),
        #                    gap(hidden, batch_index)], dim=1)

        hidden = self.lin1(hidden)

        out = self.out(hidden)

        return out
