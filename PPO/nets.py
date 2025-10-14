# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .config import DualStreamConfig


# ====== 通用模块 ======

class MLP(nn.Module):
    """简单 MLP 封装"""
    def __init__(self, in_dim, hidden_dims=(128, 128), out_dim=None, act=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        if out_dim is not None:
            layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepSetsEncoder(nn.Module):
    """
    DeepSets 风格集合编码器：
      φ: 客户端级编码
      ρ: 对客户端嵌入做汇聚后的全局上下文
    """
    def __init__(self, in_dim_cli: int, embed_dim: int = 128):
        super().__init__()
        self.phi = MLP(in_dim_cli, hidden_dims=(256, 256), out_dim=embed_dim)
        self.rho = MLP(embed_dim, hidden_dims=(256,), out_dim=embed_dim)

    def forward(self, S_cli, mask=None):
        """
        Args:
            S_cli: [B, N, D_cli]
            mask : [B, N]，1=有效，0=padding
        Returns:
            U: [B, N, E] 每客户端嵌入
            C: [B, E]     全局上下文
        """
        U = self.phi(S_cli)  # [B,N,E]
        if mask is not None:
            mask_f = mask.unsqueeze(-1)
            U_masked = U * mask_f
            C = U_masked.sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1.0))
        else:
            C = U.mean(dim=1)
        C = self.rho(C)
        return U, C


# ====== 连续动作分布：Tanh-高斯，用于剪枝率 p ======

class TanhNormal:
    """
    y = tanh(x), x ~ Normal(mu, sigma)，并映射到 [low, high]
    支持 sample() 与 log_prob()
    """

    def __init__(self, mu, log_std, low=0.1, high=0.9):
        # ---- 数值兜底（新增） ----
        mu = torch.nan_to_num(mu, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        log_std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-20.0).clamp(-20, 2)

        self.mu = mu
        self.log_std = log_std
        self.std = self.log_std.exp().clamp_min(1e-6)  # 防 0

        # 严格检查（可留着调试期）
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.std).all():
            raise RuntimeError("[TanhNormal] invalid mu/std")

        self.base = Normal(self.mu, self.std)
        self.low = low
        self.high = high

    def sample(self):
        x = self.base.rsample()         # reparam
        y = torch.tanh(x)
        p = (y + 1) / 2 * (self.high - self.low) + self.low
        # 变换修正项
        log_prob = self.base.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
        log_prob = log_prob.sum(dim=-1)
        return p, log_prob

    def log_prob(self, p):
        y = (p - self.low) / (self.high - self.low) * 2 - 1
        y = y.clamp(-0.999999, 0.999999)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh
        log_prob = self.base.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
        return log_prob.sum(dim=-1)

    @property
    def entropy(self):
        # 近似：使用 base 正态的熵和（足够稳定）
        return self.base.entropy().sum(dim=-1)


# ====== 双流 Actor-Critic ======

class DualStreamActorCritic(nn.Module):
    """
    双流（P-Stream & E-Stream）+ 门控融合 + 多头 Actor + 双价值头
    输入：
      S_glob: [B, d_glob]
      S_cli : [B, N, d_cli]
      mask  : [B, N]
    输出（每客户端）动作分布参数与价值
    """
    def __init__(self, cfg: DualStreamConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden

        # 性能流
        self.P_glob = MLP(cfg.d_glob, (H, H), H)
        self.P_cli = DeepSetsEncoder(cfg.d_cli, embed_dim=H)

        # 效率流
        self.E_glob = MLP(cfg.d_glob, (H, H), H)
        self.E_cli = DeepSetsEncoder(cfg.d_cli, embed_dim=H)

        # 门控融合：concat(zP_i, zE_i, zP_glob, zE_glob) -> 4H
        self.fuse_gate = nn.Linear(4 * H, 1)
        self.fuseP = MLP(4 * H, (H, H), H)
        self.fuseE = MLP(4 * H, (H, H), H)

        # Actor 头
        self.mu_head = nn.Linear(H, 1)
        self.logstd_head = nn.Linear(H, 1)
        self.num_E = cfg.E_max - cfg.E_min + 1
        self.E_head = nn.Linear(H, self.num_E)

        # Critic 头
        self.VP = nn.Linear(H, 1)
        self.VE = nn.Linear(H, 1)

    def forward_enc(self, S_glob, S_cli, mask=None):
        B, N, _ = S_cli.shape
        zP_g = self.P_glob(S_glob)           # [B,H]
        uP, _ = self.P_cli(S_cli, mask)      # [B,N,H]

        zE_g = self.E_glob(S_glob)
        uE, _ = self.E_cli(S_cli, mask)

        zP_gb = zP_g.unsqueeze(1).expand(B, N, -1)
        zE_gb = zE_g.unsqueeze(1).expand(B, N, -1)

        cat = torch.cat([uP, uE, zP_gb, zE_gb], dim=-1)  # [B,N,4H]
        gate = torch.sigmoid(self.fuse_gate(cat))        # [B,N,1]
        z = gate * self.fuseP(cat) + (1 - gate) * self.fuseE(cat)  # [B,N,H]
        return z

    def act_value(self, S_glob, S_cli, mask=None):
        z = self.forward_enc(S_glob, S_cli, mask)    # [B,N,H]

        mu = self.mu_head(z)
        log_std = self.logstd_head(z).clamp(-5, 2)
        logits_E = self.E_head(z)

        VP = self.VP(z)
        VE = self.VE(z)
        if self.cfg.use_value_mix:
            V = self.cfg.wP * VP + self.cfg.wE * VE
        else:
            V = 0.5 * (VP + VE)
        return mu, log_std, logits_E, V

    @torch.no_grad()
    def select_actions(self, S_glob, S_cli, mask=None):
        """
        采样动作（用于下发）
        Returns:
          p: [B,N,1] 连续剪枝率（已映射到 [p_low, p_high]）
          E: [B,N,1] 整数轮数（E_min..E_max）
          logp: [B,N] 联合 logprob
          entropy: [B,N] 两分布熵之和
        """
        mu, log_std, logits_E, _ = self.act_value(S_glob, S_cli, mask)

        tn = TanhNormal(mu, log_std, low=self.cfg.p_low, high=self.cfg.p_high)
        p, logp_p = tn.sample()

        probs_E = F.softmax(logits_E, dim=-1)
        cat = Categorical(probs=probs_E)
        E_idx = cat.sample().unsqueeze(-1)   # [B,N,1] in [0..K-1]
        E = E_idx + self.cfg.E_min
        logp_E = cat.log_prob(E_idx.squeeze(-1))
        entropy = tn.entropy + cat.entropy()

        logp = logp_p + logp_E
        return dict(p=p, E=E, logp=logp, entropy=entropy)

    def evaluate_actions(self, S_glob, S_cli, p, E, mask=None):
        """
        评估给定动作（用于 PPO 更新）
        Returns:
          logp, entropy, V
        """
        mu, log_std, logits_E, V = self.act_value(S_glob, S_cli, mask)

        # —— 数值兜底（避免 NaN/Inf） ——
        mu = torch.nan_to_num(mu, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        log_std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-20.0).clamp(-20.0, 2.0)
        logits_E = torch.nan_to_num(logits_E, nan=0.0).clamp(-20.0, 20.0)

        # 连续动作 p：TanhNormal
        tn = TanhNormal(mu, log_std, low=self.cfg.p_low, high=self.cfg.p_high)
        logp_p = tn.log_prob(p)

        # 离散动作 E：使用 logits 更稳
        cat = Categorical(logits=logits_E)
        # 将环境传回的 E ∈ [E_min, E_max] 映射到 [0, num_E-1]
        E_idx = (E - self.cfg.E_min).clamp(0, self.num_E - 1).squeeze(-1).long()
        logp_E = cat.log_prob(E_idx)

        entropy = tn.entropy + cat.entropy()
        logp = logp_p + logp_E
        return dict(logp=logp, entropy=entropy, V=V)
