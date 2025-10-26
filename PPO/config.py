# PPO/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOCommonCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4


@dataclass
class Stage1Cfg:
    # 输入：逐客户端 d1 + 全局 d_g
    d_client: int = 6  # [Tn, acc_i, Dn, H_i, p_prev, En]
    d_global: int = 6  # [T.min/mean/max/std, acc_now, N]
    hidden: int = 256
    # 离散剪枝 bins
    p_low: float = 0.0
    p_high: float = 0.5
    num_bins: int = 6
    prune_bins: List[float] = field(default_factory=list)  # If empty, auto-generate linspace between p_low and p_high with num_bins
    # 采样温度
    tau_e: float = 1.2


@dataclass
class Stage2Cfg:
    # 输入：逐客户端 d2 + 全局 d_g
    d_client: int = 6  # [Tn, H_i, Dn, acc_i, p_i, En]
    d_global: int = 6
    hidden: int = 256
    tau_e: float = 1.2
    # 轮数范围与总预算
    E_min: int = 1
    E_max: int = 19
    tau_total: int = 50  # 总预算，通常 = N * 期望均值


@dataclass
class TwoStageConfig:
    device: str = "cuda"
    common: PPOCommonCfg = field(default_factory=PPOCommonCfg)
    s1: Stage1Cfg = field(default_factory=Stage1Cfg)
    s2: Stage2Cfg = field(default_factory=Stage2Cfg)
