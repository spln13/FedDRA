# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class DualStreamConfig:
    """
    双流 PPO 的可调配置（含热身/混合/稳定化/正则的扩展超参）
    """

    # 维度设定
    d_glob: int                 # 全局特征维度
    d_cli: int                  # 单客户端特征维度
    hidden: int = 256           # 流内嵌入维度

    # 动作空间（连续剪枝率 + 离散训练轮数）
    p_low: float = 0.2          # 剪枝率最小值
    p_high: float = 0.8         # 剪枝率最大值
    E_min: int = 1              # 训练轮数最小
    E_max: int = 19              # 训练轮数最大

    # PPO 超参
    gamma: float = 0.99
    lam: float = 0.95           # GAE(lambda)
    clip_range: float = 0.10    # 更保守的剪切（冷启动期更稳）
    ent_coef: float = 1e-3      # 初期小探索
    vf_coef: float = 0.5
    kl_coef: float = 0.0
    max_grad_norm: float = 0.4
    target_kl: float = 0.01
    lr: float = 3e-5

    # 价值融合（若有双价值头）
    use_value_mix: bool = True
    wP: float = 0.5             # P-Value 权重
    wE: float = 0.5             # E-Value 权重

    # === 动作稳定（选择动作时的后处理）===
    inertia_alpha: float = 0.5  # EMA 惯性系数, 0=全旧, 1=全新
    hysteresis_eps: float = 0.05  # 滞回阈值: |Δp|<eps 则保持不变
    bucket_bins: int = 0        # 分桶数量(0=关闭；如 6/8 可大幅降抖动)

    # === 训练期的动作平滑正则（在 PPO loss 中）===
    smooth_lambda: float = 1e-2 # λ · smooth_loss(p - p_prev)
    smooth_loss: str = "huber"  # "huber" | "l2" | "l1"

    # === 热身与行为混合（规则策略 ↔ PPO）===
    warmup_rounds: int = 10             # 前K轮只收集/不更新，用规则/混合
    mix_epsilon_start: float = 1.0      # ε 起点：完全规则
    mix_epsilon_end: float = 0.0        # ε 终点：纯 PPO
    mix_epsilon_decay_rounds: int = 20  # ε 线性退火轮数

    # === 剪枝率变更的系统约束 ===
    delta_eps: float = 0.05      # 变更滞回阈值（小改动不触发/不罚）
    cooldown_rounds: int = 3     # 更换 p 的最小间隔轮数
    mask_ttl: int = 4            # 掩码复用的最少轮数（到期且超阈才重剪）

    # === 奖励中的变更代价权重 ===
    lambda_p: float = 0.5        # 在 R 中惩罚 Δp 的权重（会按区间长度归一化）

    # === 训练/采样杂项（如需要）===
    # 你也可以在 Trainer 里用这些字段控制 batch_size, update_epochs 等
    # update_epochs: int = 4
    # batch_size: int = 256