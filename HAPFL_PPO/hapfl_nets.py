# PPO/hapfl_nets.py
import torch
import torch.nn as nn


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    )


class SizeActor(nn.Module):
    def __init__(self, in_dim, hidden, num_bins):
        super().__init__()
        self.net = mlp(in_dim, hidden, num_bins)
        with torch.no_grad():
            nn.init.orthogonal_(self.net[-1].weight, 0.01)
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, X):  # X: [N, d1+d_g]
        return self.net(X)  # [N, B]


class SizeCritic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)

    def forward(self, X):  # [N, d1+d_g]
        return self.net(X)  # [N,1]


class EpochActor(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)  # 对每个 client 一个打分

    def forward(self, X):  # [N, d2+d_g]
        return self.net(X).squeeze(-1).view(1, -1)  # [1,N]


class EpochCritic(nn.Module):
    def __init__(self, d_global, hidden):
        super().__init__()
        self.net = mlp(d_global, hidden, 1)

    def forward(self, g):  # g: [K,d_g] 或 [1,d_g]
        if g.dim() == 1: g = g.view(1, -1)
        return self.net(g)  # [K,1] or [1,1]
