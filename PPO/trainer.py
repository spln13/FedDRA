# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DualStreamConfig
from .nets import DualStreamActorCritic
from .buffer import RolloutBuffer


class DualStreamPPO:
    """
    训练器封装：
      - select_actions(): 在线采样动作
      - store_transition(): 写入缓冲
      - ppo_update(): 使用缓冲数据进行一次 PPO 更新
    """
    def __init__(self, cfg: DualStreamConfig,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.cfg = cfg
        self.device = device
        self.net = DualStreamActorCritic(cfg).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer()

    # ---------- 外部接口 ----------

    @torch.no_grad()
    def select_actions(self, S_glob, S_cli, mask=None):
        """
        Args:
          S_glob: [B, d_glob]
          S_cli : [B, N, d_cli]
          mask  : [B, N]
        Returns:
          dict(p, E, logp, entropy)
        """
        S_glob = S_glob.to(self.device)
        S_cli = S_cli.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        return self.net.select_actions(S_glob, S_cli, mask)

    def store_transition(self, transition: dict):
        """
        transition 至少应包含：
          S_glob, S_cli, mask, p, E, logp, V, reward, done
        """
        item = {k: v.to(self.device) for k, v in transition.items()}
        self.buffer.add(item)

    def ppo_update(self, epochs: int = 4, batch_size: int = 64, normalize_adv: bool = True):
        """
        使用缓冲区收集的 on-policy 数据进行一次 PPO 更新
        """
        # 1) GAE / returns
        self.buffer.compute_gae_and_returns(self.cfg.gamma, self.cfg.gae_lambda)

        # 2) 打包
        S_glob = self.buffer.cat("S_glob")         # [T*B, d_glob]
        S_cli = self.buffer.cat("S_cli")           # [T*B, N, d_cli]
        mask = self.buffer.cat("mask") if "mask" in self.buffer.data[0] else None
        p = self.buffer.cat("p")                   # [T*B, N, 1]
        E = self.buffer.cat("E")                   # [T*B, N, 1]
        logp_old = self.buffer.cat("logp")         # [T*B, N]
        V_old = self.buffer.cat("V")               # [T*B, N, 1]
        R = self.buffer.returns.reshape(-1, *V_old.shape[1:])  # [T*B,N,1]
        A = (R - V_old).detach()

        if normalize_adv:
            mean, std = A.mean(), A.std().clamp_min(1e-6)
            A = (A - mean) / std

        TB, N, _ = V_old.shape

        # 展平工具
        def flat(x):
            s = x.shape
            if len(s) == 3:   # [TB,N,D]
                return x.reshape(TB * N, s[-1])
            elif len(s) == 2: # [TB,N]
                return x.reshape(TB * N)
            else:             # [TB,N,1]
                return x.reshape(TB * N, 1)

        S_glob_rep = S_glob.unsqueeze(1).expand(TB, N, -1).reshape(TB * N, -1)
        S_cli_eval = S_cli  # evaluate 时再按需 reshape

        p_f = flat(p)
        E_f = flat(E).long()
        logp_old_f = flat(logp_old)
        R_f = flat(R)
        A_f = flat(A)

        if mask is not None:
            mask_f = flat(mask)
            valid_idx = (mask_f > 0.5).nonzero(as_tuple=False).squeeze(-1)
        else:
            valid_idx = torch.arange(TB * N, device=self.device)

        # 3) 小批次多 epoch 训练
        for _ in range(epochs):
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=self.device)]
            for i in range(0, perm.numel(), batch_size):
                idx = perm[i:i+batch_size]

                Sg_b = S_glob_rep[idx]
                Sci_b = S_cli_eval.reshape(TB * N, -1)[idx].unsqueeze(1)  # [B,1,d_cli]
                mask_b = torch.ones((idx.numel(), 1), device=self.device)

                p_b = p_f[idx]
                E_b = E_f[idx].unsqueeze(-1)
                logp_old_b = logp_old_f[idx]
                R_b = R_f[idx]
                A_b = A_f[idx]

                out = self.net.evaluate_actions(Sg_b, Sci_b, p_b.unsqueeze(1), E_b, mask_b)
                logp_new = out["logp"].squeeze(-1)
                entropy = out["entropy"].squeeze(-1)
                V = out["V"].squeeze(-1)

                ratio = torch.exp(logp_new - logp_old_b)
                surr1 = ratio * A_b.squeeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * A_b.squeeze(-1)
                loss_pi = -torch.min(surr1, surr2).mean()

                # value-clip（使用近似旧值）
                V_old_b = V.detach()
                V_clipped = V_old_b + (V - V_old_b).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
                loss_v1 = (V - R_b.squeeze(-1)).pow(2)
                loss_v2 = (V_clipped - R_b.squeeze(-1)).pow(2)
                loss_v = 0.5 * torch.max(loss_v1, loss_v2).mean()

                loss_ent = -entropy.mean()

                # 可选 KL
                loss_kl = torch.tensor(0.0, device=self.device)
                if self.cfg.kl_coef > 0:
                    with torch.no_grad():
                        kl = (logp_old_b - logp_new).mean()
                    loss_kl = self.cfg.kl_coef * kl

                loss = loss_pi + self.cfg.vf_coef * loss_v + self.cfg.ent_coef * (-loss_ent) + loss_kl

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        # 4) 清空缓冲
        self.buffer.clear()
