# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


# ========== 采样辅助（沿用你原先的接口名） ==========
def sample_discrete(logits, tau=1.0):
    # logits: [B, K]
    dist = Categorical(logits=logits / max(tau, 1e-6))
    a = dist.sample()  # [B]
    logp = dist.log_prob(a)  # [B]
    ent = dist.entropy()  # [B]
    return a, logp, ent, dist


def sample_continuous(mu, std, p_min, p_max):
    # tanh-Normal 到 [p_min, p_max]
    dist = Normal(mu, std)
    x = dist.rsample()  # reparameterization
    y = torch.tanh(x)
    p = (y + 1) * 0.5 * (p_max - p_min) + p_min
    logp = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
    logp = logp.sum(dim=-1)
    ent = dist.entropy().sum(dim=-1)
    return p, logp, ent, dist


# ========== 基础 MLP ==========
def mlp(in_dim, hidden=(128, 64), out_dim=1, act=nn.Tanh):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


# ========== Stage-1 ==========
class Stage1DiscreteActor(nn.Module):
    def __init__(self, input_dim, n_bins):
        super().__init__()
        self.net = mlp(input_dim, hidden=(128, 64), out_dim=n_bins)

    def forward(self, x):
        # x: [B, input_dim]，此处 input_dim=1
        return self.net(x)


class Stage1ContActor(nn.Module):
    def __init__(self, input_dim, latent=64):
        super().__init__()
        self.backbone = mlp(input_dim, hidden=(128, 64), out_dim=latent)
        self.mu = nn.Linear(latent, 1)
        self.logstd = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = torch.tanh(self.backbone(x))
        mu = self.mu(h)
        std = torch.exp(self.logstd).expand_as(mu)
        return mu, std


class Stage1Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = mlp(input_dim, hidden=(128, 64), out_dim=1)

    def forward(self, x):
        return self.net(x)


# ========== Stage-2 ==========
class Stage2Actor(nn.Module):
    """
    给定 [B, s2_dim]（现在 s2_dim=1），输出到 k_max 的 logits；前向时根据 k 切片。
    """

    def __init__(self, input_dim, k_max=256):
        super().__init__()
        self.k_max = k_max
        self.backbone = mlp(input_dim, hidden=(128, 64), out_dim=128)
        self.head = nn.Linear(128, k_max)

    def forward(self, x, k):
        h = torch.tanh(self.backbone(x))
        logits_full = self.head(h)  # [B, k_max]
        return logits_full[:, :k]  # [B, k]


class Stage2Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = mlp(input_dim, hidden=(128, 64), out_dim=1)

    def forward(self, x):
        return self.net(x)


# nets.py 关键新增
class Stage2ActorClientwise(nn.Module):
    """
    输入: S2_cli [N, 1]（每个 client 一行，一维特征，例如 Tm' 或 Tn'）
    输出: logits [1, N]（对每个 client 的打分）
    """

    def __init__(self, in_dim=1, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 每个样本一个 logit
        )

    def forward(self, S2_cli: torch.Tensor) -> torch.Tensor:
        # S2_cli: [N, 1] -> logits [N, 1] -> [1, N]
        logits = self.mlp(S2_cli)  # [N, 1]
        return logits.transpose(0, 1)  # [1, N]
