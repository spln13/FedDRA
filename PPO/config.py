# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class PPOCommonCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    batch_size: int = 256
    # 训练稳定
    target_kl: float = 0.02


@dataclass
class Stage1Cfg:
    # —— PPO1（剪枝率/剪枝档）——
    # 你可以二选一：离散档 or 连续率
    use_discrete_bins: bool = True
    prune_bins: tuple = (0.2, 0.35, 0.5, 0.65, 0.8)  # 离散剪枝档
    p_min: float = 0.2
    p_max: float = 0.8
    s1_dim: int = 6  # 输入维度（根据你构造的 s1[i] 决定）
    # 探索温度（仅离散时有效）
    tau_e: float = 1.2


@dataclass
class Stage2Cfg:
    # —— PPO2（E 分配）——
    s2_dim: int = 6  # 输入维度（根据你构造的 s2[i] 或 s2_global 决定）
    # 对参与的 k 个客户端做softmax分配
    # 你可以固定总预算 tau_total，再有：E[i] = round(softmax_i * tau_total)
    tau_total: int = 10
    tau_e: float = 1.2  # softmax温度（调大更平均，调小更尖锐）


@dataclass
class TwoStageConfig:
    device: str = "cuda"
    common: PPOCommonCfg = PPOCommonCfg()
    s1: Stage1Cfg = Stage1Cfg()
    s2: Stage2Cfg = Stage2Cfg()
