# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical, Normal


def _mlp(in_dim, hidden=(128, 128), out_dim=None, act=nn.ReLU):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    if out_dim is not None:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


# =============== PPO1: 剪枝率/剪枝档 ===============
class Stage1DiscreteActor(nn.Module):
    """离散剪枝档（bins）"""

    def __init__(self, s_dim, n_bins):
        super().__init__()
        self.body = _mlp(s_dim, (256, 256), n_bins)

    def forward(self, s):
        return self.body(s)  # logits


class Stage1ContActor(nn.Module):
    """连续剪枝率（tanh → [p_min, p_max]）"""

    def __init__(self, s_dim):
        super().__init__()
        self.mu = _mlp(s_dim, (256, 256), 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, s):
        mu = torch.tanh(self.mu(s))  # [-1,1]
        std = self.log_std.exp().clamp(1e-4, 2.0)
        return mu, std


class Stage1Critic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.v = _mlp(s_dim, (256, 256), 1)

    def forward(self, s):
        return self.v(s)


# =============== PPO2: 轮数分配 (softmax over clients) ===============
class Stage2Actor(nn.Module):
    """
    输入：你可以用 “每客户端特征拼接后+全局特征” 的向量 s2_global
    输出：对 k 个客户端的打分 logits[k]
    """

    def __init__(self, s_dim, k_max=256):
        super().__init__()
        self.body = _mlp(s_dim, (256, 256), k_max)  # 你在 forward 时 slice 到实际 k

    def forward(self, s_global, k):
        logits = self.body(s_global)[:, :k]  # [B, k]
        return logits


class Stage2Critic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.v = _mlp(s_dim, (256, 256), 1)

    def forward(self, s):
        return self.v(s)


# =============== 采样/评估的工具函数 ===============
def sample_discrete(logits, tau=1.0):
    logits = (logits / max(tau, 1e-6))
    logits = torch.nan_to_num(logits, nan=0.0).clamp(-40, 40)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    dist = Categorical(logits=logits)
    a = dist.sample()
    logp = dist.log_prob(a)
    ent = dist.entropy()
    return a, logp, ent, dist


def sample_continuous(mu, std, p_min, p_max):
    dist = Normal(mu, std)
    x = dist.rsample()  # reparam
    y = torch.tanh(x)  # [-1,1]
    p = (y + 1) / 2 * (p_max - p_min) + p_min
    logp = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
    logp = logp.sum(dim=-1)
    ent = dist.entropy().sum(dim=-1)
    return p, logp, ent, dist
