# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOCommonCfg:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    target_kl: float = 0.03


@dataclass
class Stage1Cfg:
    # === 关键：PPO1 输入维度改为 1 ===
    s1_dim: int = 1
    use_discrete_bins: bool = True
    prune_bins: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2])
    # 连续动作时的边界（若 use_discrete_bins=False 时使用）
    p_min: float = 0.0
    p_max: float = 0.5
    # 采样温度
    tau_e: float = 1.0


# config.py 关键处
@dataclass
class Stage2Cfg:
    s2_dim: int = 1              # ← 由 6 改为 1（每个 client 的 Tm 或 Tn）
    tau_total: int = 64
    tau_e: float = 1.0
    use_clientwise: bool = True  # ← 新增：S2 按客户端逐条输入


@dataclass
class TwoStageConfig:
    device: str = "cuda"
    common: PPOCommonCfg = PPOCommonCfg()
    s1: Stage1Cfg = Stage1Cfg(s1_dim=1)
    s2: Stage2Cfg = Stage2Cfg(s2_dim=1, use_clientwise=True)
    p_low: float = 0.0
    p_high: float = 0.5
    E_min: int = 1
    E_max: int = 19