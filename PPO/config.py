# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOCommonCfg:
    lr: float = 1e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    target_kl: float = 0.03
    adaptive_kl: bool = True
    kl_low_ratio: float = 0.35
    kl_high_ratio: float = 1.5
    lr_scale_up: float = 1.08
    lr_scale_down: float = 0.85
    lr_scale_min: float = 0.4
    lr_scale_max: float = 3.0


@dataclass
class Stage1Cfg:
    # 每个客户端的 Stage-1 特征维度
    s1_dim: int = 4
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
    # use_clientwise=True 时，表示“每客户端特征维度”
    s2_dim: int = 4
    tau_total: int = 64
    tau_e: float = 1.0
    use_clientwise: bool = True


@dataclass
class TwoStageConfig:
    device: str = "cuda"
    common: PPOCommonCfg = field(default_factory=PPOCommonCfg)
    s1: Stage1Cfg = field(default_factory=Stage1Cfg)
    s2: Stage2Cfg = field(default_factory=Stage2Cfg)
    p_low: float = 0.0
    p_high: float = 0.5
    E_min: int = 1
    E_max: int = 19
