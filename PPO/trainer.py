# PPO/trainer.py
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .config import TwoStageConfig
from .buffer import SimpleTrajBuffer
from .nets import Stage1Actor, Stage1Critic, Stage2Actor, Stage2Critic


def _reduce_global_g(g: torch.Tensor) -> torch.Tensor:
    """
    buffer.stack() 之后 g 可能是 [K, d_g]；这里把它还原成 [d_g]（取最近一条）。
    """
    if g.dim() == 2:
        return g[-1]
    return g


class _PPOCore:
    def __init__(self, actor: nn.Module, critic: nn.Module, cfg_common, device: str):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg_common
        self.device = device
        params = list(actor.parameters()) + list(critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg_common.lr)

    def _compute_adv(self, r, v, v_next, done):
        # 这里按照样本独立（T=1）的简化版 GAE：直接用 delta 归一化
        with torch.no_grad():
            delta = r + self.cfg.gamma * (1 - done) * v_next - v
            adv = (delta - delta.mean()) / (delta.std() + 1e-8)
            ret = adv + v
        return adv, ret


class TwoStagePPO:
    """
    两阶段 PPO：
      Stage-1: 逐客户端离散动作 -> 剪枝率 bins
      Stage-2: 逐客户端打分 -> softmax 分配 τ_total 得到 E（critic-only 稳定版）
    """

    def __init__(self, cfg: TwoStageConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # ====== Stage-1 ======
        self.s1_actor = Stage1Actor(cfg.s1.d_client, cfg.s1.d_global, cfg.s1.num_bins, cfg.s1.hidden).to(self.device)
        self.s1_critic = Stage1Critic(cfg.s1.d_client, cfg.s1.d_global, cfg.s1.hidden).to(self.device)
        self.ppo1 = _PPOCore(self.s1_actor, self.s1_critic, cfg.common, str(self.device))
        # 剪枝 bins（若 cfg.s1.prune_bins 为空，则用 linspace 生成）
        if getattr(cfg.s1, "prune_bins", None) and len(cfg.s1.prune_bins) > 0:
            self.prune_bins = torch.tensor(cfg.s1.prune_bins, dtype=torch.float32, device=self.device)
        else:
            self.prune_bins = torch.linspace(cfg.s1.p_low, cfg.s1.p_high, cfg.s1.num_bins, device=self.device)

        # ====== Stage-2 ======
        self.s2_actor = Stage2Actor(cfg.s2.d_client, cfg.s2.d_global, cfg.s2.hidden).to(self.device)
        self.s2_critic = Stage2Critic(cfg.s2.d_global, cfg.s2.hidden).to(self.device)
        self.ppo2 = _PPOCore(self.s2_actor, self.s2_critic, cfg.common, str(self.device))

        # buffers
        self.buf1 = SimpleTrajBuffer(device=str(self.device))
        self.buf2 = SimpleTrajBuffer(device=str(self.device))

    # ---------------- Stage-1: select/store/update ----------------
    @torch.no_grad()
    def select_pruning(self, S1_cli: torch.Tensor, g: torch.Tensor, deterministic: bool = False) -> Dict[
        str, torch.Tensor]:
        """
        S1_cli: [N, d1] 逐客户端特征
        g     : [d_g] 或 [K, d_g] 全局上下文（内部会归约到 [d_g] 使用）
        """
        g_eff = _reduce_global_g(g)
        logits = self.s1_actor(S1_cli, g_eff)  # [N, B]
        tau = self.cfg.s1.tau_e
        probs = torch.softmax(logits / tau, dim=-1)
        dist = Categorical(probs=probs)
        a = torch.argmax(probs, dim=-1) if deterministic else dist.sample()  # [N]
        logp = dist.log_prob(a)  # [N]
        v = self.s1_critic(S1_cli, g_eff)  # [N,1]
        p = self.prune_bins[a].unsqueeze(-1)  # [N,1]
        return dict(p=p, a=a, logp=logp, v=v, probs=probs, logits=logits)

    def store_transition_stage1(self,
                                s: tuple, a: torch.Tensor, logp: torch.Tensor, v: torch.Tensor,
                                r: torch.Tensor, s_next: tuple, done: Optional[torch.Tensor] = None):
        """
        s:      (S1_cli, g)  -> ([N,d1], [d_g] or [K,d_g])
        a:      [N]          离散 bin 索引
        logp:   [N]
        v:      [N,1]
        r:      [N,1]        逐客户端奖励向量
        s_next: (S1_cli', g')
        done:   [N,1]        通常全0
        """
        if done is None:
            done = torch.zeros_like(r)
        self.buf1.add(s=s, a=a, logp=logp, v=v, r=r, s_next=s_next, done=done)

    def ppo_update_stage1(self):
        if len(self.buf1) == 0:
            return {}
        traj = self.buf1.stack()
        S1_cli, g = traj["s"]
        a = traj["a"].long()
        logp_old = traj["logp"]
        v_old = traj["v"]
        r = traj["r"]
        S1_cli2, g2 = traj["s_next"]
        done = traj["done"]

        g = _reduce_global_g(g)
        g2 = _reduce_global_g(g2)

        with torch.no_grad():
            v_next = self.s1_critic(S1_cli2, g2)
        adv, ret = self.ppo1._compute_adv(r, v_old, v_next, done)

        logits = self.s1_actor(S1_cli, g)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = (logp - logp_old).exp()

        cfgc = self.cfg.common
        adv_use = adv.squeeze(-1)
        surr1 = ratio * adv_use
        surr2 = torch.clamp(ratio, 1 - cfgc.clip_coef, 1 + cfgc.clip_coef) * adv_use
        actor_loss = -torch.min(surr1, surr2).mean() - cfgc.ent_coef * dist.entropy().mean()

        v = self.s1_critic(S1_cli, g)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfgc.vf_coef * critic_loss
        self.ppo1.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.s1_actor.parameters()) + list(self.s1_critic.parameters()),
                                 cfgc.max_grad_norm)
        self.ppo1.opt.step()

        self.buf1.clear()
        approx_kl = (logp_old - logp).mean().clamp_min(0.0).item()
        return dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl)

    # ---------------- Stage-2: select/store/update ----------------
    @torch.no_grad()
    def select_epochs(self,
                      S2_cli: torch.Tensor, g: torch.Tensor,
                      tau_total: int, E_min: int, E_max: int,
                      deterministic: bool = False) -> Dict[str, Any]:
        """
        S2_cli:   [N, d2] 逐客户端特征（内部会与 g 智能合并）
        g:        [d_g] 或 [K,d_g]
        tau_total: int    总预算（轮数总和）
        """
        g_eff = _reduce_global_g(g)
        logits = self.s2_actor(S2_cli, g_eff)  # [1, N]
        tau = self.cfg.s2.tau_e
        probs = torch.softmax(logits / tau, dim=-1)  # [1,N]

        alloc = (probs[0] * float(max(tau_total, 1))).cpu().numpy()  # [N]
        E = np.clip(np.round(alloc), E_min, E_max).astype(int)
        diff = int(tau_total - int(E.sum()))
        if diff != 0:
            order = np.argsort(-probs[0].cpu().numpy()) if diff > 0 else np.argsort(probs[0].cpu().numpy())
            for idx in order:
                if diff == 0:
                    break
                cand = E[idx] + (1 if diff > 0 else -1)
                if E_min <= cand <= E_max:
                    E[idx] = cand
                    diff += (-1 if diff > 0 else +1)

        v = self.s2_critic(g_eff)  # [1,1]
        return dict(E=E, probs=probs, logits=logits, v=v)

    def store_transition_stage2(self,
                                s: torch.Tensor, logits: torch.Tensor, v: torch.Tensor,
                                r: torch.Tensor, s_next: torch.Tensor, done: Optional[torch.Tensor] = None):
        """
        s:       g 或者 (S2_cli, g) 均可，这里用 g（[1,d_g] 或 [K,d_g]）
        logits:  [1,N]   （保留以便需要时推断 N；当前 update 不依赖）
        v:       [1,1]
        r:       [1,1] 全局奖励
        s_next:  [1,d_g] 或 [K,d_g]
        done:    [1,1]
        """
        if done is None:
            done = torch.zeros_like(v)
        self.buf2.add(s=s, a_logits=logits, v=v, r=r, s_next=s_next, done=done)

    def ppo_update_stage2(self):
        if len(self.buf2) == 0:
            return {}
        traj = self.buf2.stack()
        g = traj["s"]
        v_old = traj["v"]
        r = traj["r"]
        g2 = traj["s_next"]
        done = traj["done"]

        g = _reduce_global_g(g)
        g2 = _reduce_global_g(g2)

        with torch.no_grad():
            v_next = self.s2_critic(g2)
        adv, ret = self.ppo2._compute_adv(r, v_old, v_next, done)

        v = self.s2_critic(g)
        critic_loss = F.mse_loss(v, ret)

        self.ppo2.opt.zero_grad()
        (self.cfg.common.vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(list(self.s2_critic.parameters()), self.cfg.common.max_grad_norm)
        self.ppo2.opt.step()

        self.buf2.clear()
        return dict(loss=critic_loss.item(), actor=0.0, critic=critic_loss.item(), kl=0.0)
