# PPO/nets.py
import torch
import torch.nn as nn


# ===== Stage-1: 剪枝率打分（逐客户端 + 全局） =====
class Stage1Actor(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, num_bins: int, hidden: int = 256):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(in_dim_client + in_dim_global, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_bins)
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        # S_cli: [N, d1], g: [d_g]
        N = S_cli.size(0)
        g_exp = g.view(1, -1).expand(N, -1)
        x = torch.cat([S_cli, g_exp], dim=-1)
        logits = self.net(x)  # [N, num_bins]
        return logits


class Stage1Critic(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim_client + in_dim_global, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        N = S_cli.size(0)
        g_exp = g.view(1, -1).expand(N, -1)
        x = torch.cat([S_cli, g_exp], dim=-1)
        V = self.net(x)  # [N,1]
        return V


# ===== Stage-2: 轮数打分（逐客户端 + 全局） =====
class Stage2Actor(nn.Module):
    def __init__(self, in_dim_client: int, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim_client + in_dim_global, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # 每个 client 一个 logit
        )

    def forward(self, S_cli: torch.Tensor, g: torch.Tensor):
        # S_cli: [N, d2], g: [d_g]
        N = S_cli.size(0)
        g_exp = g.view(1, -1).expand(N, -1)
        x = torch.cat([S_cli, g_exp], dim=-1)
        logits = self.net(x).squeeze(-1)  # [N]
        return logits.view(1, N)  # [1, N] (与旧代码兼容)


class Stage2Critic(nn.Module):
    def __init__(self, in_dim_global: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim_global, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, g: torch.Tensor):
        # g: [d_g] 或 [B, d_g]；统一成 [B, d_g]
        if g.dim() == 1:
            g = g.view(1, -1)
        V = self.net(g)  # [B,1]
        return V
