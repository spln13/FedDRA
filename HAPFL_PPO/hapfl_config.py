# PPO/hapfl_config.py
from dataclasses import dataclass, field


@dataclass
class PPOCommon:
    gamma: float = 0.99
    lam: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass
class SizeHeadCfg:
    d_client: int = 6  # 例：[Tn, acc_i, Dn, H_i, p_prev, En]
    d_global: int = 6  # 例：[Tmin, Tmean, Tmax, Tstd, acc_now, N]
    hidden: int = 256
    # 剪枝率离散 bin（HAPFL 原是“模型大小”离散；这里用“剪枝率”离散）
    p_low: float = 0.10
    p_high: float = 0.50
    num_bins: int = 6
    tau_e: float = 1.0  # 动作温度


@dataclass
class EpochHeadCfg:
    d_client: int = 6  # 例：[Tn, H_i, Dn, acc_i, p_next, En]
    d_global: int = 6
    hidden: int = 256
    tau_e: float = 1.0
    E_min: int = 1
    E_max: int = 19
    tau_total: int = 50  # 全局预算 = N * 期望均值


@dataclass
class HapflFedDRACfg:
    device: str = "cuda"
    common: PPOCommon = field(default_factory=PPOCommon)
    size: SizeHeadCfg = field(default_factory=SizeHeadCfg)
    epoch: EpochHeadCfg = field(default_factory=EpochHeadCfg)
