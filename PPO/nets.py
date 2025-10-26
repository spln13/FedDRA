# PPO/nets.py
import torch
import torch.nn as nn


def _expand_g_to_batch(g: torch.Tensor, N: int) -> torch.Tensor:
    """把 g 扩展成 [N, d_g]；支持 g 是 [d_g] 或 [K, d_g]（此时取最后一条）。"""
    if g.dim() == 2:
        g = g[-1]  # 取最近一条全局上下文
    return g.view(1, -1).expand(N, -1)  # [N, d_g]


def _merge_cli_g(S_cli: torch.Tensor, g: torch.Tensor,
                 expect_cli: int, expect_g: int) -> torch.Tensor:
    """
    智能合并：
      - 如果 S_cli 已经是 [N, expect_cli+expect_g]，认为 g 已拼过，直接返回；
      - 否则把 g（[d_g] 或 [K,d_g]）扩到 [N,d_g] 后再拼一次。
    """
    N, D = S_cli.shape
    if D == expect_cli + expect_g:
        return S_cli
    g_exp = _expand_g_to_batch(g, N)
    return torch.cat([S_cli, g_exp], dim=-1)


# ===== Stage-1: 剪枝率打分（逐客户端 + 全局） =====
class Stage1Actor(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, num_bins: int, hidden: int = 256):
        super().__init__()
        self.num_bins = num_bins
        self.in_dim_client = in_dim_client
        self.in_dim_global = in_dim_global
        self.net = nn.Sequential(
            nn.LazyLinear(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_bins)
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        x = _merge_cli_g(S_cli, g, self.in_dim_client, self.in_dim_global)  # [N, d1+d_g]
        logits = self.net(x)  # [N, num_bins]
        return logits


class Stage1Critic(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.in_dim_client = in_dim_client
        self.in_dim_global = in_dim_global
        self.net = nn.Sequential(
            nn.LazyLinear(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        x = _merge_cli_g(S_cli, g, self.in_dim_client, self.in_dim_global)  # [N, d1+d_g]
        V = self.net(x)  # [N,1]
        return V


# ===== Stage-2: 轮数打分（逐客户端 + 全局） =====
class Stage2Actor(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.in_dim_client = in_dim_client
        self.in_dim_global = in_dim_global
        self.net = nn.Sequential(
            nn.LazyLinear(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # 每个 client 一个 logit
        )

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        # S_cli: [N, d2] 或 [N, d2+d_g]（若已预拼则不会重复拼）
        x = _merge_cli_g(S_cli, g, self.in_dim_client, self.in_dim_global)  # [N, d2+d_g]
        logits = self.net(x).squeeze(-1)  # [N]
        return logits.view(1, logits.size(0))  # [1, N]


class Stage2Critic(nn.Module):
    def __init__(self, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim_global, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, g: torch.Tensor):
        # g: [d_g] 或 [B, d_g]；统一到 [B, d_g]
        if g.dim() == 1:
            g = g.view(1, -1)
        return self.net(g)  # [B,1]
