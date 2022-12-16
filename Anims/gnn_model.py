import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, GraphConv


class CBRP(nn.Module):
    def __init__(self, n_in, n_out, n_k, st, pad, pool=2):
        super(CBRP, self).__init__()
        self.conv = nn.Conv1d(n_in, n_out, n_k, st, padding=pad)
        self.bn = nn.BatchNorm1d(n_out, momentum=0.5)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        return self.pool(F.leaky_relu(self.bn(self.conv(x))))


class SkipC(nn.Module):
    def __init__(self, n_ch, n_k):
        assert n_k % 2
        super(SkipC, self).__init__()
        self.cbrp = CBRP(n_ch, n_ch, n_k, 1, n_k // 2, pool=1)

    def forward(self, x):
        return x + self.cbrp(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Sequential(
            CBRP(1, 32, 9, 4, 0, 1),
            SkipC(32, 5),
            SkipC(32, 5),
            SkipC(32, 5),
            CBRP(32, 64, 5, 2, 0, 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gcn1 = GraphConv(64, 64)
        self.bn1 = BatchNorm(64, momentum=0.5)
        self.gcn2 = GraphConv(64, 64)
        self.bn2 = BatchNorm(64, momentum=0.5)
        self.gcn3 = GraphConv(64, 64)
        self.bn3 = BatchNorm(64, momentum=0.5)
        self.gcn4 = GraphConv(64, 64)
        self.bn4 = BatchNorm(64, momentum=0.5)
        self.gcn5 = GraphConv(64, 64)
        self.bn5 = BatchNorm(64, momentum=0.5)
        self.linear = nn.Linear(64, 2)

    def forward(self, x, e):
        B, ND, L = x.shape
        x = x.view(B * ND, 1, L)
        x = self.cnn1(x)
        x = x.view(B * ND, 64)
        es = torch.cat([e + 492 * i for i in range(B)], dim=1)

        x = x + F.leaky_relu(self.bn1(self.gcn1(x, es)))
        x = x + F.leaky_relu(self.bn2(self.gcn2(x, es)))
        x = x + F.leaky_relu(self.bn3(self.gcn3(x, es)))
        x = x + F.leaky_relu(self.bn4(self.gcn4(x, es)))
        x = F.leaky_relu(self.bn5(self.gcn5(x, es)))

        y = x.view(B, ND, 64)
        y = torch.mean(y, dim=1)
        y = self.linear(y)
        return y
