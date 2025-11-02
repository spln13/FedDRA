# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .config import TwoStageConfig
from .buffer import PPORolloutBuffer
from .nets import (
    Stage1DiscreteActor, Stage1ContActor, Stage1Critic,
    Stage2Actor, Stage2Critic,
    sample_discrete, sample_continuous
)


class _PPOCore:
    def __init__(self, actor, critic, cfg, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=cfg.lr
        )

    def _compute_adv(self, r, v, v_next, done):
        """
        r, v, v_next, done: [T, *tail]，tail 可以是 [1] 或 [B,1] 等
        返回 adv, ret，形状与 v 相同
        """
        r = r.float(); v = v.float(); v_next = v_next.float(); done = done.float()
        T = r.shape[0]
        tail_shape = tuple(v.shape[1:])

        adv = torch.zeros((T,)+tail_shape, device=r.device, dtype=r.dtype)
        gae = torch.zeros(tail_shape, device=r.device, dtype=r.dtype)

        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        with torch.no_grad():
            for t in reversed(range(T)):
                delta_t = r[t] + gamma * (1.0 - done[t]) * v_next[t] - v[t]
                gae = delta_t + gamma * lam * (1.0 - done[t]) * gae
                adv[t] = gae

            ret = adv + v
            # 全局归一化 advantage
            flat = adv.view(T, -1)
            flat = (flat - flat.mean()) / (flat.std() + 1e-8)
            adv = flat.view_as(adv)
        return adv, ret

    def update_discrete(self, traj):
        cfg = self.cfg
        s = traj["s"]
        a = traj["a"].long()
        logp_old = traj["logp"]
        v_old = traj["v"]
        r = traj["r"]
        s_next = traj["s_next"]
        done = traj["done"]

        with torch.no_grad():
            v_next = self.critic(s_next)
        adv, ret = self._compute_adv(r, v_old, v_next, done)

        logits = self.actor(s)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = (logp - logp_old).exp()

        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * dist.entropy().mean()

        v = self.critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfg.vf_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                 cfg.max_grad_norm)
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(loss=loss.item(), actor=actor_loss.item(),
                    critic=critic_loss.item(), kl=approx_kl.item())

    def update_continuous(self, traj, p_min, p_max):
        cfg = self.cfg
        s = traj["s"]
        a = traj["a"]
        logp_old = traj["logp"]
        v_old = traj["v"]
        r = traj["r"]
        s_next = traj["s_next"]
        done = traj["done"]

        with torch.no_grad():
            v_next = self.critic(s_next)
        adv, ret = self._compute_adv(r, v_old, v_next, done)

        mu, std = self.actor(s)

        # 反推 tanh-normal 原变量近似，重算 logp
        y = (a - p_min) / (p_max - p_min) * 2 - 1
        y = y.clamp(-0.999999, 0.999999)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))
        dist = Normal(mu, std)
        logp = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
        logp = logp.sum(dim=-1)

        ratio = (logp - logp_old).exp()
        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * dist.entropy().sum(dim=-1).mean()

        v = self.critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfg.vf_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                 cfg.max_grad_norm)
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(loss=loss.item(), actor=actor_loss.item(),
                    critic=critic_loss.item(), kl=approx_kl.item())


class TwoStagePPO:
    """
    管理两套 PPO：
      - PPO1: 剪枝率 / 剪枝档（支持离散bins或连续区间）
      - PPO2: E 分配（HAPFL式：对 k 个客户端打分 softmax，再按 tau_total 比例分配）
    对外暴露：
      select_pruning(s1_batch) -> {'p','a','logp','v'}
      select_epochs(s2_global, k) -> {'E','probs','v','logits'}
      select_epochs_per_client(S2_cli, g=None, ...) -> 同上（逐客户端输入，内部聚合）
      store_transition_stage1 / ppo_update_stage1
      store_transition_stage2 / ppo_update_stage2
      store_transition_stage2_per_client / ppo_update_stage2_per_client
    """

    def __init__(self, cfg: TwoStageConfig):
        self.cfg = cfg
        dev = cfg.device

        # === PPO1 ===
        if cfg.s1.use_discrete_bins:
            self.s1_actor = Stage1DiscreteActor(cfg.s1.s1_dim, len(cfg.s1.prune_bins))
            self.s1_is_discrete = True
        else:
            self.s1_actor = Stage1ContActor(cfg.s1.s1_dim)
            self.s1_is_discrete = False
        self.s1_critic = Stage1Critic(cfg.s1.s1_dim)
        self.ppo1 = _PPOCore(self.s1_actor, self.s1_critic, cfg.common, device=dev)

        # === PPO2 ===（k-way softmax）
        self.k_max = 256
        self.s2_actor = Stage2Actor(cfg.s2.s2_dim, k_max=self.k_max)
        self.s2_critic = Stage2Critic(cfg.s2.s2_dim)
        self.ppo2 = _PPOCore(self.s2_actor, self.s2_critic, cfg.common, device=dev)

        # buffers
        self.buf1 = PPORolloutBuffer(device=dev)
        self.buf2 = PPORolloutBuffer(device=dev)      # 全局式
        self.buf2c = PPORolloutBuffer(device=dev)     # “逐客户端输入后聚合”的存储（与 buf2 等价用法）

        self.device = dev
        self.prune_bins = torch.tensor(cfg.s1.prune_bins, dtype=torch.float32, device=dev)

    # ---------------- PPO1: pruning ----------------
    @torch.no_grad()
    def select_pruning(self, s1_batch):
        """
        s1_batch: [B, d_s1]，每行=一个 client 的状态
        return dict(p=[B,1], a, logp=[B], v=[B,1])
        """
        s1 = s1_batch.to(self.device).float()
        v = self.s1_critic(s1)
        if self.s1_is_discrete:
            logits = self.s1_actor(s1)
            a, logp, _, _ = sample_discrete(logits, tau=self.cfg.s1.tau_e)
            p = self.prune_bins[a].unsqueeze(-1)  # [B,1]
        else:
            mu, std = self.s1_actor(s1)
            p, logp, _, _ = sample_continuous(mu, std, self.cfg.s1.p_min, self.cfg.s1.p_max)
            a = p
        return dict(p=p, a=a, logp=logp, v=v)

    # ---------------- PPO2: epochs (global k-way) ----------------
    @torch.no_grad()
    def select_epochs(self, s2_global, k):
        """
        s2_global: [B, d_s2]（建议 B=1）
        k: 客户端个数
        return dict(E=[k], probs=[k], v, logits=[k])
        """
        s2 = s2_global.to(self.device).float()
        v = self.s2_critic(s2)
        logits = self.s2_actor(s2, k)                  # [B,k]
        logits = logits / max(self.cfg.s2.tau_e, 1e-6)
        dist = Categorical(logits=logits)
        probs = dist.probs.squeeze(0)                  # [k]
        sigma = probs / probs.sum()
        E = torch.round(sigma * self.cfg.s2.tau_total).long().clamp(min=1)  # [k]
        return dict(E=E, probs=probs, v=v, logits=logits.squeeze(0))

    @torch.no_grad()
    def select_epochs_per_client(self, S2_cli, g=None, E_min=None, E_max=None, aggregate="default"):
        """
        逐客户端输入 S2_cli: [N, d_cli]，内部做聚合统计 -> g: [1, d_s2]，复用 select_epochs
        NOTE: 这是与现有 Stage2Actor 兼容的“包一层”的接口，便于 server 侧调用统一。
        """
        N = S2_cli.shape[0]
        s2_dim = self.cfg.s2.s2_dim
        device = self.device

        if g is None:
            X = S2_cli.detach().clone().to(device).float()
            # 约定第0列是单位时延特征 Tn，若无则用全1
            Tn = X[:, 0] if X.shape[1] >= 1 else torch.ones(N, device=device)
            # 约定第3列是 acc_i（若无则退化为最后一列）
            acc_col = 3 if X.shape[1] > 3 else (X.shape[1] - 1)
            acc_i = X[:, acc_col]

            # 聚合：min/mean/max/std(Tn) + mean(acc) + N
            t_min = Tn.min().unsqueeze(0)
            t_mean = Tn.mean().unsqueeze(0)
            t_max = Tn.max().unsqueeze(0)
            t_std = Tn.std(unbiased=False).unsqueeze(0)
            a_mean = acc_i.mean().unsqueeze(0)
            n_val = torch.tensor([float(N)], device=device)

            g = torch.cat([t_min, t_mean, t_max, t_std, a_mean, n_val], dim=0).view(1, -1)  # [1,6]
            if g.shape[1] < s2_dim:
                pad = torch.zeros(1, s2_dim - g.shape[1], device=device)
                g = torch.cat([g, pad], dim=1)
            elif g.shape[1] > s2_dim:
                g = g[:, :s2_dim]

        out = self.select_epochs(g, k=N)
        E = out["E"]
        # 做硬边界裁剪
        E_low = int(self.cfg.E_min) if E_min is None else int(E_min)
        E_high = int(self.cfg.E_max) if E_max is None else int(E_max)
        E = E.clamp(min=E_low, max=E_high)
        return dict(E=E, probs=out["probs"], v=out["v"], logits=out["logits"])

    # ---------------- Buffers (Stage-1 / Stage-2 global / Stage-2 per-client wrapper) -------------
    def store_transition_stage1(self, s, a, logp, v, r, s_next, done):
        self.buf1.add(
            s=s, a=a, logp=logp, v=v,
            r=torch.tensor([r], device=self.device),
            s_next=s_next, done=torch.tensor([done], device=self.device)
        )

    def store_transition_stage2(self, s, a_logits, v, r, s_next, done):
        """
        全局式：把 actor 的 logits（[1,k]）与 k 一起写入，便于 update 重建维度
        r: 标量
        """
        with torch.no_grad():
            probs = torch.softmax(a_logits, dim=-1)   # [1,k]
            a = torch.argmax(probs, dim=-1)           # [1]
            logp = torch.log(probs[0, a])             # [1]
        k_tensor = torch.tensor([a_logits.shape[-1]], device=self.device, dtype=torch.long)
        self.buf2.add(
            s=s, a=a, logp=logp, v=v,
            r=torch.tensor([r], device=self.device),
            s_next=s_next, done=torch.tensor([done], device=self.device),
            a_logits=a_logits.detach(), k=k_tensor
        )

    def store_transition_stage2_per_client(self, S2_cli, r_vec, g=None, done=0):
        """
        逐客户端接口（当前为“聚合后仍用全局式”的实现）：
          - S2_cli: [N, d_cli]
          - r_vec : [N] 或 [N,1]（逐客户端奖励）
        做法：先把 S2_cli -> g（与 select_epochs_per_client 相同的聚合），
              再把 r_vec 做均值（或你可改成加权）得到标量 r，写入 buf2c。
        """
        out = self.select_epochs_per_client(S2_cli, g=g)  # 构造/对齐 g，但不真正分配 E
        g_used = out["v"].new_zeros((1, self.cfg.s2.s2_dim))  # 只占位；update 时只需要 s/s_next 的形状
        # 为了一致性，用 g 作为 s/s_next
        if g is None:
            g = self._aggregate_S2_cli(S2_cli)

        # 统一 r 为标量
        r_vec = torch.as_tensor(r_vec, device=self.device).float().view(-1)
        r_scalar = r_vec.mean().item()

        # dummy logits: 用 N 维全零占位
        N = S2_cli.shape[0]
        a_logits = torch.zeros(1, N, device=self.device)
        k_tensor = torch.tensor([N], device=self.device, dtype=torch.long)

        # 与 store_transition_stage2 一致的字段
        with torch.no_grad():
            probs = torch.softmax(a_logits, dim=-1)
            a = torch.argmax(probs, dim=-1)
            logp = torch.log(probs[0, a])

        self.buf2c.add(
            s=g, a=a, logp=logp, v=torch.zeros(1,1,device=self.device),
            r=torch.tensor([r_scalar], device=self.device),
            s_next=g, done=torch.tensor([done], device=self.device),
            a_logits=a_logits, k=k_tensor
        )

    def _aggregate_S2_cli(self, S2_cli):
        """与 select_epochs_per_client 中的聚合保持一致，返回 g: [1, d_s2]"""
        N = S2_cli.shape[0]
        s2_dim = self.cfg.s2.s2_dim
        device = self.device
        X = S2_cli.detach().clone().to(device).float()
        Tn = X[:, 0] if X.shape[1] >= 1 else torch.ones(N, device=device)
        acc_col = 3 if X.shape[1] > 3 else (X.shape[1] - 1)
        acc_i = X[:, acc_col]
        t_min = Tn.min().unsqueeze(0)
        t_mean = Tn.mean().unsqueeze(0)
        t_max = Tn.max().unsqueeze(0)
        t_std = Tn.std(unbiased=False).unsqueeze(0)
        a_mean = acc_i.mean().unsqueeze(0)
        n_val = torch.tensor([float(N)], device=device)
        g = torch.cat([t_min, t_mean, t_max, t_std, a_mean, n_val], dim=0).view(1, -1)
        if g.shape[1] < s2_dim:
            pad = torch.zeros(1, s2_dim - g.shape[1], device=device)
            g = torch.cat([g, pad], dim=1)
        elif g.shape[1] > s2_dim:
            g = g[:, :s2_dim]
        return g

    # ---------------- Updates ----------------
    def ppo_update_stage1(self):
        if len(self.buf1) == 0:
            return {}
        traj = self.buf1.stack()
        outs = {}
        for _ in range(self.cfg.common.update_epochs):
            if self.s1_is_discrete:
                out = self.ppo1.update_discrete(traj)
            else:
                out = self.ppo1.update_continuous(traj, self.cfg.s1.p_min, self.cfg.s1.p_max)
            outs = out
            if out["kl"] > self.cfg.common.target_kl:
                break
        self.buf1.clear()
        return outs

    def ppo_update_stage2(self):
        """
        全局式更新：使用 buf2（与当前 server 写入对应）
        """
        if len(self.buf2) == 0:
            return {}
        traj = self.buf2.stack()
        s = traj["s"]; a = traj["a"].long(); logp_old = traj["logp"]
        v_old = traj["v"]; r = traj["r"]; s_next = traj["s_next"]; done = traj["done"]

        # 取出 k
        if "k" in traj:
            k = int(traj["k"][0].item())
        elif "a_logits" in traj:
            k = int(traj["a_logits"].shape[-1])
        else:
            raise RuntimeError("Need 'k' or 'a_logits' in buffer to infer number of clients (k).")

        with torch.no_grad():
            v_next = self.s2_critic(s_next)
        adv, ret = self.ppo2._compute_adv(r, v_old, v_next, done)

        logits = self.s2_actor(s, k)                   # [B,k]
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = (logp - logp_old).exp()

        cfgc = self.cfg.common
        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - cfgc.clip_coef, 1 + cfgc.clip_coef) * adv.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean() - cfgc.ent_coef * dist.entropy().mean()

        v = self.s2_critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfgc.vf_coef * critic_loss
        self.ppo2.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.s2_actor.parameters()) + list(self.s2_critic.parameters()),
                                 cfgc.max_grad_norm)
        self.ppo2.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0).item()
        outs = dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl)
        self.buf2.clear()
        return outs

    def ppo_update_stage2_per_client(self):
        """
        与 ppo_update_stage2 等价（目前“逐客户端奖励->聚合标量奖励”的实现）。
        预留该入口，后续如果换成“每客户端对 {E_min..E_max} 分类”的 actor，这里直接改前向与损失即可。
        """
        if len(self.buf2c) == 0:
            return {}
        traj = self.buf2c.stack()
        s = traj["s"]; a = traj["a"].long(); logp_old = traj["logp"]
        v_old = traj["v"]; r = traj["r"]; s_next = traj["s_next"]; done = traj["done"]

        if "k" in traj:
            k = int(traj["k"][0].item())
        elif "a_logits" in traj:
            k = int(traj["a_logits"].shape[-1])
        else:
            raise RuntimeError("Need 'k' or 'a_logits' in buffer to infer number of clients (k).")

        with torch.no_grad():
            v_next = self.s2_critic(s_next)
        adv, ret = self.ppo2._compute_adv(r, v_old, v_next, done)

        logits = self.s2_actor(s, k)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = (logp - logp_old).exp()

        cfgc = self.cfg.common
        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - cfgc.clip_coef, 1 + cfgc.clip_coef) * adv.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean() - cfgc.ent_coef * dist.entropy().mean()

        v = self.s2_critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfgc.vf_coef * critic_loss
        self.ppo2.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.s2_actor.parameters()) + list(self.s2_critic.parameters()),
                                 cfgc.max_grad_norm)
        self.ppo2.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0).item()
        outs = dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl)
        self.buf2c.clear()
        return outs