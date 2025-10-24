# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .config import TwoStageConfig
from .buffer import PPORolloutBuffer
from .nets import (
    Stage1DiscreteActor, Stage1ContActor, Stage1Critic,
    Stage2Actor, Stage2Critic,
    sample_discrete, sample_continuous
)


class _PPOCore:
    def __init__(self, actor, critic, cfg, device="cpu", is_discrete=True):
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
        r, v, v_next, done: [T, *tail]
        tail 可以是 [1] 或 [B,1] 等；本函数自适应。
        返回:
        adv, ret: 与 v 同形
        """
        cfg = self.cfg
        r      = r.float()
        v      = v.float()
        v_next = v_next.float()
        done   = done.float()

        T = r.shape[0]
        tail_shape = tuple(v.shape[1:])  # e.g. (1,) or (B,1)

        adv = torch.zeros((T,)+tail_shape, device=r.device, dtype=r.dtype)
        gae = torch.zeros(tail_shape, device=r.device, dtype=r.dtype)

        gamma, lam = cfg.gamma, cfg.gae_lambda
        with torch.no_grad():
            for t in reversed(range(T)):
                delta_t = r[t] + gamma * (1.0 - done[t]) * v_next[t] - v[t]
                gae = delta_t + gamma * lam * (1.0 - done[t]) * gae
                adv[t] = gae

            ret = adv + v

            # 归一化 advantage（对所有元素做一次全局归一化）
            adv_flat = adv.view(T, -1)
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            adv = adv_flat.view_as(adv)

        return adv, ret

    def update_discrete(self, traj):
        cfg = self.cfg
        s = traj["s"];
        a = traj["a"].long();
        logp_old = traj["logp"];
        v_old = traj["v"]
        r = traj["r"];
        s_next = traj["s_next"];
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
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl.item())

    def update_continuous(self, traj, p_min, p_max):
        cfg = self.cfg
        s = traj["s"];
        a = traj["a"];
        logp_old = traj["logp"];
        v_old = traj["v"]
        r = traj["r"];
        s_next = traj["s_next"];
        done = traj["done"]

        with torch.no_grad():
            v_next = self.critic(s_next)
        adv, ret = self._compute_adv(r, v_old, v_next, done)

        mu, std = self.actor(s)
        # 反推 tanh-normal 的原变量近似（直接用 a 当输入的近似）
        # 我们用重算 logp 的方式：先把 a 映射回 tanh 空间：
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
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl.item())


class TwoStagePPO:
    """
    统一管理两套 PPO：
      - PPO1: 剪枝率/剪枝档
      - PPO2: E 分配（softmax over clients）
    对外暴露：
      select_pruning(s1_batch) -> p_target 或 bin_idx
      select_epochs(s2_global, k) -> E_vec 长度为 k
      store_transition_stage1(...); update_stage1(...)
      store_transition_stage2(...); update_stage2(...)
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

        # === PPO2 ===
        # 让 Actor 输出至多 k_max=256 的 logits，你在 forward 时切到实际 k
        self.k_max = 256
        self.s2_actor = Stage2Actor(cfg.s2.s2_dim, k_max=self.k_max)
        self.s2_critic = Stage2Critic(cfg.s2.s2_dim)
        self.ppo2 = _PPOCore(self.s2_actor, self.s2_critic, cfg.common, device=dev)

        # buffers
        self.buf1 = PPORolloutBuffer(device=dev)
        self.buf2 = PPORolloutBuffer(device=dev)

        self.device = dev
        self.prune_bins = torch.tensor(cfg.s1.prune_bins, dtype=torch.float32, device=dev)

    @torch.no_grad()
    def select_pruning(self, s1_batch):
        """
        输入：s1_batch [B, d_s1]，每行=一个client的状态（归一化后的 T_d 等）
        输出：
           dict: { 'p': [B,1], 'a': 动作张量, 'logp': [B,], 'v': [B,1] }
        """
        s1 = s1_batch.to(self.device).float()
        v = self.s1_critic(s1)
        if self.s1_is_discrete:
            logits = self.s1_actor(s1)
            a, logp, ent, _ = sample_discrete(logits, tau=self.cfg.s1.tau_e)
            p = self.prune_bins[a].unsqueeze(-1)  # [B,1]
        else:
            mu, std = self.s1_actor(s1)
            p, logp, ent, _ = sample_continuous(mu, std, self.cfg.s1.p_min, self.cfg.s1.p_max)
            a = p  # 连续动作本身就是p
        return dict(p=p, a=a, logp=logp, v=v)

    @torch.no_grad()
    def select_epochs(self, s2_global, k):
        """
        输入：s2_global [B, d_s2]（建议把全局统计/拼接特征做成一行B=1）
             k: 本轮参与的客户端个数
        输出：E_vec 长度 k（long）
        """
        s2 = s2_global.to(self.device).float()
        v = self.s2_critic(s2)
        logits = self.s2_actor(s2, k)  # [B, k]
        logits = logits / max(self.cfg.s2.tau_e, 1e-6)
        dist = Categorical(logits=logits)
        # 直接取概率做 softmax 权重，而不是采样一个索引
        probs = dist.probs.squeeze(0)  # [k]
        # 分配 E
        sigma = probs / probs.sum()
        E = torch.round(sigma * self.cfg.s2.tau_total).long().clamp(min=1)  # [k]
        return dict(E=E, probs=probs, v=v, logits=logits.squeeze(0))

    # ========== 存轨迹（每轮一次即可） ==========
    def store_transition_stage1(self, s, a, logp, v, r, s_next, done):
        self.buf1.add(s=s, a=a, logp=logp, v=v, r=torch.tensor([r], device=self.device),
                      s_next=s_next, done=torch.tensor([done], device=self.device))

    def store_transition_stage2(self, s, a_logits, v, r, s_next, done):
        """
        a_logits: [1, k] 对本轮 k 个客户端的打分logits（配合 softmax 分配E）
        这里把 a_logits 也存进 buffer（或至少把 k 存进去），便于 update 阶段重建 actor 的输出维度。
        """
        with torch.no_grad():
            probs = torch.softmax(a_logits, dim=-1)  # [1,k]
            a = torch.argmax(probs, dim=-1)          # [1]
            logp = torch.log(probs[0, a])            # [1]

        # ✅ 关键：把 a_logits 以及 k 存起来
        k_tensor = torch.tensor([a_logits.shape[-1]], device=self.device, dtype=torch.long)

        self.buf2.add(
            s=s, a=a, logp=logp, v=v,
            r=torch.tensor([r], device=self.device),
            s_next=s_next, done=torch.tensor([done], device=self.device),
            a_logits=a_logits.detach(),   # <—— 新增
            k=k_tensor                    # <—— 新增（冗余但安全）
        )
    # ========== 更新 ==========
    def ppo_update_stage1(self):
        if len(self.buf1) == 0: return {}
        traj = self.buf1.stack()
        # 多轮update
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
        if len(self.buf2) == 0:
            return {}

        traj = self.buf2.stack()
        s       = traj["s"]
        a       = traj["a"].long()
        logp_old= traj["logp"]
        v_old   = traj["v"]
        r       = traj["r"]
        s_next  = traj["s_next"]
        done    = traj["done"]

        # ✅ 关键：稳妥获取 k
        if "k" in traj:
            k = int(traj["k"][0].item())
        elif "a_logits" in traj:
            k = int(traj["a_logits"].shape[-1])
        else:
            raise RuntimeError("Need 'k' or 'a_logits' in buffer to infer number of clients (k).")

        with torch.no_grad():
            v_next = self.s2_critic(s_next)
        adv, ret = self.ppo2._compute_adv(r, v_old, v_next, done)

        logits = self.s2_actor(s, k)  # <<<<<< 需要 k
        dist   = torch.distributions.Categorical(logits=logits)
        logp   = dist.log_prob(a)
        ratio  = (logp - logp_old).exp()

        cfgc = self.cfg.common
        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - cfgc.clip_coef, 1 + cfgc.clip_coef) * adv.squeeze(-1)
        actor_loss  = -torch.min(surr1, surr2).mean() - cfgc.ent_coef * dist.entropy().mean()

        v = self.s2_critic(s)
        critic_loss = torch.nn.functional.mse_loss(v, ret)

        loss = actor_loss + cfgc.vf_coef * critic_loss
        self.ppo2.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.s2_actor.parameters()) + list(self.s2_critic.parameters()),
            cfgc.max_grad_norm
        )
        self.ppo2.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0).item()
        outs = dict(loss=loss.item(), actor=actor_loss.item(), critic=critic_loss.item(), kl=approx_kl)

        self.buf2.clear()
        return outs