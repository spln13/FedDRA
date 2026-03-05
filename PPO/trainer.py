# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial, Normal

from .buffer import PPORolloutBuffer
from .config import TwoStageConfig
from .nets import (
    Stage1ContActor,
    Stage1Critic,
    Stage1DiscreteActor,
    Stage2Actor,
    Stage2ActorClientwise,
    Stage2Critic,
    sample_continuous,
    sample_discrete,
)


class _PPOCore:
    def __init__(self, actor, critic, cfg, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=cfg.lr,
        )

    def _compute_adv(self, r, v, v_next, done):
        """
        r, v, v_next, done: [T, *tail]
        return: adv, ret with same shape as v
        """
        r = r.float()
        v = v.float()
        v_next = v_next.float()
        done = done.float()
        T = r.shape[0]
        tail_shape = tuple(v.shape[1:])

        adv = torch.zeros((T,) + tail_shape, device=r.device, dtype=r.dtype)
        gae = torch.zeros(tail_shape, device=r.device, dtype=r.dtype)

        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        with torch.no_grad():
            for t in reversed(range(T)):
                delta_t = r[t] + gamma * (1.0 - done[t]) * v_next[t] - v[t]
                gae = delta_t + gamma * lam * (1.0 - done[t]) * gae
                adv[t] = gae

            ret = adv + v
            flat = adv.reshape(T, -1)
            flat = (flat - flat.mean()) / (flat.std() + 1e-8)
            adv = flat.reshape_as(adv)
        return adv, ret

    def update_discrete(self, traj, temperature=1.0):
        cfg = self.cfg
        s = traj["s"]
        a = traj["a"].long()
        logp_old = traj["logp"].float()
        v_old = traj["v"]
        r = traj["r"]
        s_next = traj["s_next"]
        done = traj["done"]

        with torch.no_grad():
            v_next = self.critic(s_next)
        adv, ret = self._compute_adv(r, v_old, v_next, done)

        logits = self.actor(s)
        dist = Categorical(logits=logits / max(float(temperature), 1e-6))
        logp = dist.log_prob(a)
        logp_old = logp_old.view_as(logp)
        ratio = (logp - logp_old).exp()

        adv_term = adv.squeeze(-1)
        if adv_term.dim() < ratio.dim():
            adv_term = adv_term.expand_as(ratio)
        surr1 = ratio * adv_term
        surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv_term
        actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * dist.entropy().mean()

        v = self.critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfg.vf_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            cfg.max_grad_norm,
        )
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(
            loss=loss.item(),
            actor=actor_loss.item(),
            critic=critic_loss.item(),
            kl=approx_kl.item(),
        )

    def update_continuous(self, traj, p_min, p_max):
        cfg = self.cfg
        s = traj["s"]
        a = traj["a"]
        logp_old = traj["logp"].float()
        v_old = traj["v"]
        r = traj["r"]
        s_next = traj["s_next"]
        done = traj["done"]

        with torch.no_grad():
            v_next = self.critic(s_next)
        adv, ret = self._compute_adv(r, v_old, v_next, done)

        mu, std = self.actor(s)
        y = (a - p_min) / (p_max - p_min) * 2 - 1
        y = y.clamp(-0.999999, 0.999999)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))
        dist = Normal(mu, std)
        logp = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-8)
        logp = logp.sum(dim=-1)
        logp_old = logp_old.view_as(logp)

        ratio = (logp - logp_old).exp()
        adv_term = adv.squeeze(-1)
        if adv_term.dim() < ratio.dim():
            adv_term = adv_term.expand_as(ratio)
        surr1 = ratio * adv_term
        surr2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv_term
        actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * dist.entropy().sum(dim=-1).mean()

        v = self.critic(s)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + cfg.vf_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            cfg.max_grad_norm,
        )
        self.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0)
        return dict(
            loss=loss.item(),
            actor=actor_loss.item(),
            critic=critic_loss.item(),
            kl=approx_kl.item(),
        )


class TwoStagePPO:
    """
    PPO1: pruning-rate policy (per-client action).
    PPO2: epoch-allocation policy (multinomial allocation over clients).
    """

    def __init__(self, cfg: TwoStageConfig):
        self.cfg = cfg
        dev = cfg.device

        # PPO1
        if cfg.s1.use_discrete_bins:
            self.s1_actor = Stage1DiscreteActor(cfg.s1.s1_dim, len(cfg.s1.prune_bins))
            self.s1_is_discrete = True
        else:
            self.s1_actor = Stage1ContActor(cfg.s1.s1_dim)
            self.s1_is_discrete = False
        self.s1_critic = Stage1Critic(cfg.s1.s1_dim)
        self.ppo1 = _PPOCore(self.s1_actor, self.s1_critic, cfg.common, device=dev)

        # PPO2
        self.s2_use_clientwise = bool(getattr(cfg.s2, "use_clientwise", False))
        self.k_max = 256
        if self.s2_use_clientwise:
            self.s2_actor = Stage2ActorClientwise(in_dim=cfg.s2.s2_dim, hidden=128)
        else:
            self.s2_actor = Stage2Actor(cfg.s2.s2_dim, k_max=self.k_max)
        self.s2_critic = Stage2Critic(cfg.s2.s2_dim)
        self.ppo2 = _PPOCore(self.s2_actor, self.s2_critic, cfg.common, device=dev)

        self.buf1 = PPORolloutBuffer(device=dev)
        self.buf2 = PPORolloutBuffer(device=dev)

        self.device = dev
        self.prune_bins = torch.tensor(cfg.s1.prune_bins, dtype=torch.float32, device=dev)

    # ---------------- stage1 ----------------
    @torch.no_grad()
    def select_pruning(self, s1_batch):
        s1 = s1_batch.to(self.device).float()
        v = self.s1_critic(s1)
        if self.s1_is_discrete:
            logits = self.s1_actor(s1)
            a, logp, _, _ = sample_discrete(logits, tau=self.cfg.s1.tau_e)
            p = self.prune_bins[a].unsqueeze(-1)
            return dict(p=p, a=a, logp=logp, v=v, logits=logits)
        mu, std = self.s1_actor(s1)
        p, logp, _, _ = sample_continuous(mu, std, self.cfg.s1.p_min, self.cfg.s1.p_max)
        return dict(p=p, a=p, logp=logp, v=v, logits=None)

    # ---------------- stage2 helpers ----------------
    def _s2_actor_logits(self, s2_state, k):
        if self.s2_use_clientwise:
            logits = self.s2_actor(s2_state)
            if logits.shape[-1] < k:
                raise ValueError(f"stage2 logits has {logits.shape[-1]} dims, but k={k}")
            return logits[..., :k]
        return self.s2_actor(s2_state, k)

    def _s2_critic_input(self, s2_state):
        if not self.s2_use_clientwise:
            return s2_state
        if s2_state.dim() == 2:  # [N, d]
            return s2_state.mean(dim=0, keepdim=True)  # [1, d]
        if s2_state.dim() == 3:  # [T, N, d]
            return s2_state.mean(dim=1)  # [T, d]
        raise ValueError(f"Unsupported stage2 state shape: {tuple(s2_state.shape)}")

    def _s2_value(self, s2_state):
        return self.s2_critic(self._s2_critic_input(s2_state))

    @staticmethod
    def _stage2_budget(k, e_min, e_max, tau_total):
        tau_total = max(int(tau_total), int(k) * int(e_min))
        residual_budget = max(0, tau_total - int(k) * int(e_min))
        cap_per_client = max(0, int(e_max) - int(e_min))
        max_residual = int(k) * cap_per_client
        residual_budget = min(residual_budget, max_residual)
        return residual_budget, cap_per_client

    @staticmethod
    def _repair_counts(counts, probs, budget, cap_per_client):
        counts = counts.clone().long()
        if budget <= 0 or cap_per_client <= 0:
            return torch.zeros_like(counts)

        counts = counts.clamp(min=0, max=cap_per_client)
        diff = int(budget - int(counts.sum().item()))
        if diff == 0:
            return counts

        if diff > 0:
            order = torch.argsort(probs, descending=True).tolist()
            for idx in order:
                if diff <= 0:
                    break
                room = cap_per_client - int(counts[idx].item())
                if room <= 0:
                    continue
                step = min(room, diff)
                counts[idx] += step
                diff -= step
        else:
            order = torch.argsort(probs, descending=False).tolist()
            need = -diff
            for idx in order:
                if need <= 0:
                    break
                can_drop = int(counts[idx].item())
                if can_drop <= 0:
                    continue
                step = min(can_drop, need)
                counts[idx] -= step
                need -= step
        return counts

    # ---------------- stage2 ----------------
    @torch.no_grad()
    def select_epochs(self, s2_state, k=None, E_min=None, E_max=None):
        s2 = s2_state.to(self.device).float()
        if k is None:
            if self.s2_use_clientwise and s2.dim() == 2:
                k = int(s2.shape[0])
            else:
                raise ValueError("k is required when stage2 input is not [N, d].")

        e_min = int(self.cfg.E_min if E_min is None else E_min)
        e_max = int(self.cfg.E_max if E_max is None else E_max)
        if e_max < e_min:
            e_max = e_min

        v = self._s2_value(s2)
        logits = self._s2_actor_logits(s2, k) / max(float(self.cfg.s2.tau_e), 1e-6)
        if logits.dim() != 2 or logits.shape[0] != 1:
            raise ValueError(f"select_epochs expects actor logits [1, k], got {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [k]
        residual_budget, cap_per_client = self._stage2_budget(k, e_min, e_max, self.cfg.s2.tau_total)

        if residual_budget <= 0:
            counts = torch.zeros(k, device=self.device, dtype=torch.long)
            logp = torch.zeros((), device=self.device)
        else:
            dist = Multinomial(total_count=int(residual_budget), probs=probs)
            raw_counts = dist.sample().long()
            counts = self._repair_counts(raw_counts, probs, residual_budget, cap_per_client)
            logp = dist.log_prob(counts.float())

        E = counts + e_min
        return dict(
            E=E.long(),
            a=counts.long(),
            logp=logp,
            v=v,
            probs=probs,
            logits=logits.squeeze(0),
            residual_budget=int(residual_budget),
        )

    @torch.no_grad()
    def select_epochs_per_client(self, S2_cli, g=None, E_min=None, E_max=None, aggregate="default"):
        del g, aggregate
        return self.select_epochs(S2_cli, k=int(S2_cli.shape[0]), E_min=E_min, E_max=E_max)

    # ---------------- buffers ----------------
    def store_transition_stage1(self, s, a, logp, v, r, s_next, done):
        self.buf1.add(
            s=s,
            a=a,
            logp=logp,
            v=v,
            r=torch.tensor([r], device=self.device),
            s_next=s_next,
            done=torch.tensor([done], device=self.device),
        )

    def store_transition_stage2(self, s, a, logp, v, r, s_next, done, residual_budget):
        a_tensor = a if torch.is_tensor(a) else torch.as_tensor(a, device=self.device)
        logp_tensor = logp if torch.is_tensor(logp) else torch.as_tensor(logp, device=self.device)
        k_tensor = torch.tensor([int(a_tensor.numel())], device=self.device, dtype=torch.long)
        budget_tensor = torch.tensor([int(residual_budget)], device=self.device, dtype=torch.long)
        self.buf2.add(
            s=s,
            a=a_tensor.long(),
            logp=logp_tensor.float(),
            v=v,
            r=torch.tensor([r], device=self.device),
            s_next=s_next,
            done=torch.tensor([done], device=self.device),
            k=k_tensor,
            budget=budget_tensor,
        )

    def store_transition_stage2_per_client(self, S2_cli, r_vec, g=None, done=0):
        del g
        out = self.select_epochs_per_client(S2_cli, E_min=self.cfg.E_min, E_max=self.cfg.E_max)
        r_vec = torch.as_tensor(r_vec, device=self.device).float().view(-1)
        r_scalar = float(r_vec.mean().item())
        self.store_transition_stage2(
            s=S2_cli,
            a=out["a"],
            logp=out["logp"],
            v=out["v"],
            r=r_scalar,
            s_next=S2_cli,
            done=done,
            residual_budget=out["residual_budget"],
        )

    # ---------------- updates ----------------
    def ppo_update_stage1(self):
        if len(self.buf1) == 0:
            return {}
        traj = self.buf1.stack()
        outs = {}
        for _ in range(self.cfg.common.update_epochs):
            if self.s1_is_discrete:
                out = self.ppo1.update_discrete(traj, temperature=self.cfg.s1.tau_e)
            else:
                out = self.ppo1.update_continuous(traj, self.cfg.s1.p_min, self.cfg.s1.p_max)
            outs = out
            if out["kl"] > self.cfg.common.target_kl:
                break
        self.buf1.clear()
        return outs

    def _ppo_update_stage2_once(self, traj):
        s = traj["s"].float()
        a = traj["a"].float()
        T = s.shape[0]
        logp_old = traj["logp"].float().view(T)
        v_old = traj["v"].float().view(T, -1)[:, :1]
        r = traj["r"].float().view(T, 1)
        s_next = traj["s_next"].float()
        done = traj["done"].float().view(T, 1)

        if "k" not in traj or "budget" not in traj:
            raise RuntimeError("Stage2 buffer must contain both 'k' and 'budget'.")
        k = int(traj["k"][0].item())
        budget = int(traj["budget"][0].item())

        with torch.no_grad():
            v_next = self._s2_value(s_next).view(T, 1)
        adv, ret = self.ppo2._compute_adv(r, v_old, v_next, done)

        logits = self._s2_actor_logits(s, k) / max(float(self.cfg.s2.tau_e), 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if budget <= 0:
            logp = torch.zeros_like(logp_old)
            ratio = torch.ones_like(logp_old)
            entropy = torch.zeros_like(logp_old)
        else:
            dist = Multinomial(total_count=budget, probs=probs)
            logp = dist.log_prob(a)
            ratio = (logp - logp_old).exp()
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)

        adv_term = adv.squeeze(-1)
        if adv_term.dim() > 1:
            adv_term = adv_term.mean(dim=-1)

        if budget <= 0:
            actor_loss = torch.zeros((), device=self.device)
        else:
            surr1 = ratio * adv_term
            surr2 = torch.clamp(ratio, 1 - self.cfg.common.clip_coef, 1 + self.cfg.common.clip_coef) * adv_term
            actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.common.ent_coef * entropy.mean()

        v = self._s2_value(s).view(T, 1)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + self.cfg.common.vf_coef * critic_loss
        self.ppo2.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.s2_actor.parameters()) + list(self.s2_critic.parameters()),
            self.cfg.common.max_grad_norm,
        )
        self.ppo2.opt.step()

        approx_kl = (logp_old - logp).mean().clamp_min(0.0).item()
        return dict(
            loss=loss.item(),
            actor=actor_loss.item(),
            critic=critic_loss.item(),
            kl=approx_kl,
        )

    def ppo_update_stage2(self):
        if len(self.buf2) == 0:
            return {}
        traj = self.buf2.stack()
        outs = {}
        for _ in range(self.cfg.common.update_epochs):
            out = self._ppo_update_stage2_once(traj)
            outs = out
            if out["kl"] > self.cfg.common.target_kl:
                break
        self.buf2.clear()
        return outs

    def ppo_update_stage2_per_client(self):
        return self.ppo_update_stage2()
