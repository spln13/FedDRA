# PPO/config.py
from dataclasses import dataclass, field


@dataclass
class PPOCommonCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    batch_size: int = 256
    target_kl: float = 0.02


@dataclass
class Stage1Cfg:
    # —— PPO1（剪枝率/剪枝档）——
    use_discrete_bins: bool = True
    prune_bins: tuple = (0.2, 0.35, 0.5, 0.65, 0.8)  # 离散剪枝档（元组是不可变的，OK）
    p_min: float = 0.2
    p_max: float = 0.8
    s1_dim: int = 6
    tau_e: float = 1.4  # 离散温度


@dataclass
class Stage2Cfg:
    # —— PPO2（E 分配）——
    s2_dim: int = 6
    tau_total: int = 10  # 每轮总训练轮数预算
    tau_e: float = 1.2  # softmax 温度


@dataclass
class TwoStageConfig:
    device: str = "cuda"
    common: PPOCommonCfg = field(default_factory=PPOCommonCfg)
    s1: Stage1Cfg = field(default_factory=Stage1Cfg)
    s2: Stage2Cfg = field(default_factory=Stage2Cfg)
    # 兼容 server 里规则策略与边界读取
    p_low: float = 0.2  # 最低剪枝率
    p_high: float = 0.9  # 最高剪枝率
    E_min: int = 1
    E_max: int = 19
