# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class DualStreamConfig:
    """
    双流 PPO 的可调配置（含超参数）
    """
    # 维度设定
    d_glob: int  # 全局特征维度
    d_cli: int  # 单客户端特征维度
    hidden: int = 256  # 流内嵌入维度

    # 动作空间
    p_low: float = 0.2  # 剪枝率最小值
    p_high: float = 0.9  # 剪枝率最大值
    E_min: int = 1  # 本地最小训练轮数
    E_max: int = 5  # 本地最大训练轮数

    # PPO 超参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    # 价值融合
    use_value_mix: bool = True
    wP: float = 0.5  # P-Value 权重
    wE: float = 0.5  # E-Value 权重
