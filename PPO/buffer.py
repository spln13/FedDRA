# -*- coding: utf-8 -*-
import torch

class RolloutBuffer:
    """
    存放 on-policy 批次数据，并计算 GAE/Returns
    要求按时间顺序 add()。
    """
    def __init__(self):
        self.data = []
        self.returns = None
        self.advs = None

    def add(self, item):
        """
        至少包含键：
          S_glob, S_cli, mask, p, E, logp, V, reward, done
        形状示例：
          S_glob: [B, d_glob]
          S_cli : [B, N, d_cli]
          mask  : [B, N]
          p     : [B, N, 1]
          E     : [B, N, 1]
          logp  : [B, N]
          V     : [B, N, 1]
          reward: [B, 1] 或 [B]
          done  : [B, 1]
        """
        self.data.append(item)

    def cat(self, key):
        return torch.cat([d[key] for d in self.data], dim=0)

    def compute_gae_and_returns(self, gamma: float, lam: float):
        """
        计算 GAE 与 returns（按时间回溯）
        """
        rewards = [d["reward"] for d in self.data]   # 每步 [B,1] 或 [B]
        values = [d["V"] for d in self.data]         # 每步 [B,N,1]
        masks_done = [1.0 - d["done"].float() for d in self.data]  # [B,1]

        B, N, _ = values[0].shape
        # 将 reward 扩展到 [B,N,1]
        rew_t = []
        for r in rewards:
            if r.dim() == 1:
                r = r.view(-1, 1, 1).expand(B, N, 1)
            elif r.dim() == 2:
                r = r.view(B, 1, 1).expand(B, N, 1)
            rew_t.append(r)

        adv_next = torch.zeros_like(values[0])
        V_next = values[-1].detach()
        returns = []

        for t in reversed(range(len(values))):
            mask_t = masks_done[t].view(B, 1, 1).expand(B, N, 1)
            delta = rew_t[t] + gamma * V_next * mask_t - values[t]
            adv_next = delta + gamma * lam * mask_t * adv_next
            ret_t = (adv_next + values[t]).detach()
            returns.insert(0, ret_t)
            V_next = values[t].detach()

        self.returns = torch.stack(returns, dim=0)   # [T,B,N,1]
        self.advs = self.returns - torch.stack(values, dim=0)

    def clear(self):
        self.data.clear()
        self.returns = None
        self.advs = None
