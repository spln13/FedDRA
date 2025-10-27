# PPO/hapfl_trainer.py
from typing import Dict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from hapfl_config import HapflFedDRACfg
from hapfl_buffer import TrajBuf
from hapfl_nets import SizeActor, SizeCritic, EpochActor, EpochCritic


class _PPO:
    def __init__(self, actor: nn.Module, critic: nn.Module, cfg):
        self.actor, self.critic, self.cfg = actor, critic, cfg
        self.opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)

    def gae(self, r, v, v_next, done):
        # r,v,v_next,done: [T,1] 或 [N,1]
        gamma, lam = self.cfg.gamma, self.cfg.lam
        with torch.no_grad():
            delta = r + gamma * (1 - done) * v_next - v
            adv = torch.zeros_like(r)
            gae = 0.0
            for t in reversed(range(len(r))):
                gae = delta[t] + gamma * lam * (1 - done[t]) * gae
                adv[t] = gae
            ret = adv + v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret


class HapflFedDRAAgent:
    def __init__(self, cfg: HapflFedDRACfg):
        self.cfg = cfg
        dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = dev

        d1 = cfg.size.d_client + cfg.size.d_global
        d2 = cfg.epoch.d_client + cfg.epoch.d_global

        self.size_actor = SizeActor(d1, cfg.size.hidden, cfg.size.num_bins).to(dev)
        self.size_critic = SizeCritic(d1, cfg.size.hidden).to(dev)
        self.ppo_size = _PPO(self.size_actor, self.size_critic, cfg.common)

        self.epoch_actor = EpochActor(d2, cfg.epoch.hidden).to(dev)
        self.epoch_critic = EpochCritic(cfg.epoch.d_global, cfg.epoch.hidden).to(dev)
        self.ppo_epoch = _PPO(self.epoch_actor, self.epoch_critic, cfg.common)

        # bins for pruning-rate (HAPFL 原是“模型大小” bins)
        self.prune_bins = torch.linspace(cfg.size.p_low, cfg.size.p_high, cfg.size.num_bins, device=dev)

        self.buf_size = TrajBuf(device=str(dev))
        self.buf_epoch = TrajBuf(device=str(dev))

    # ---------- Size head ----------
    @torch.no_grad()
    def select_pruning(self, X1: torch.Tensor, deterministic=False) -> Dict[str, torch.Tensor]:
        # X1: [N, d1]
        logits = self.size_actor(X1)
        probs = torch.softmax(logits / self.cfg.size.tau_e, dim=-1)
        dist = Categorical(probs=probs)
        a = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        v = self.size_critic(X1)
        p = self.prune_bins[a].unsqueeze(-1)  # [N,1]
        return dict(p=p, a=a, logp=logp, v=v, logits=logits, probs=probs)

    def store_size(self, X1, a, logp, v, r, X1_next, done=None):
        if done is None: done = torch.zeros_like(r)
        self.buf_size.add(s=X1, a=a.long(), logp=logp, v=v, r=r, s_next=X1_next, done=done)

    def update_size(self):
        if len(self.buf_size) == 0: return {}
        traj = self.buf_size.stack()
        X, a, logp_old, v_old, r, Xn, done = traj["s"], traj["a"], traj["logp"], traj["v"], traj["r"], traj["s_next"], \
        traj["done"]
        with torch.no_grad():
            v_next = self.size_critic(Xn)
        adv, ret = self.ppo_size.gae(r, v_old, v_next, done)

        logits = self.size_actor(X)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = (logp - logp_old).exp()

        c = self.cfg.common
        surr1 = ratio * adv.squeeze(-1)
        surr2 = torch.clamp(ratio, 1 - c.clip_coef, 1 + c.clip_coef) * adv.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean() - c.ent_coef * dist.entropy().mean()

        v = self.size_critic(X)
        critic_loss = F.mse_loss(v, ret)

        loss = actor_loss + c.vf_coef * critic_loss
        self.ppo_size.opt.zero_grad();
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.size_actor.parameters()) + list(self.size_critic.parameters()),
                                 c.max_grad_norm)
        self.ppo_size.opt.step()
        self.buf_size.clear()
        return {"loss": loss.item(), "actor": actor_loss.item(), "critic": critic_loss.item()}

    # ---------- Epoch head ----------
    @torch.no_grad()
    def select_epochs(self, X2: torch.Tensor, g: torch.Tensor,
                      tau_total: int, E_min: int, E_max: int, deterministic=False):
        logits = self.epoch_actor(X2)  # [1,N]
        probs = torch.softmax(logits / self.cfg.epoch.tau_e, dim=-1)
        alloc = (probs[0] * float(max(tau_total, 1))).cpu().numpy()
        E = np.clip(np.round(alloc), E_min, E_max).astype(int)

        # 调整满足总预算
        diff = int(tau_total - int(E.sum()))
        if diff != 0:
            pref = np.argsort(-probs[0].cpu().numpy()) if diff > 0 else np.argsort(probs[0].cpu().numpy())
            for idx in pref:
                if diff == 0: break
                cand = E[idx] + (1 if diff > 0 else -1)
                if E_min <= cand <= E_max:
                    E[idx] = cand
                    diff += (-1 if diff > 0 else +1)

        v = self.epoch_critic(g if g.dim() == 2 else g.view(1, -1))
        return dict(E=E, probs=probs, logits=logits, v=v)

    def store_epoch(self, g, logits, v, r, g_next, done=None):
        if done is None: done = torch.zeros_like(v)
        self.buf_epoch.add(s=g if g.dim() == 2 else g.view(1, -1),
                           a_logits=logits, v=v, r=r,
                           s_next=g_next if g_next.dim() == 2 else g_next.view(1, -1),
                           done=done)

    def update_epoch(self):
        if len(self.buf_epoch) == 0: return {}
        traj = self.buf_epoch.stack()
        g, v_old, r, g2, done = traj["s"], traj["v"], traj["r"], traj["s_next"], traj["done"]
        if g.dim() == 1:  g = g.unsqueeze(0)
        if g2.dim() == 1: g2 = g2.unsqueeze(0)
        if v_old.shape[0] == 1 and g.shape[0] > 1:
            v_old = v_old.expand(g.shape[0], -1)

        with torch.no_grad():
            v_next = self.epoch_critic(g2)
        adv, ret = self.ppo_epoch.gae(r, v_old, v_next, done)

        v = self.epoch_critic(g)
        c = self.cfg.common
        critic_loss = F.mse_loss(v, ret)
        self.ppo_epoch.opt.zero_grad()
        (c.vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(list(self.epoch_critic.parameters()), c.max_grad_norm)
        self.ppo_epoch.opt.step()
        self.buf_epoch.clear()
        return {"loss": critic_loss.item(), "actor": 0.0, "critic": critic_loss.item()}
