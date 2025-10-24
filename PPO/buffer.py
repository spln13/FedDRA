# -*- coding: utf-8 -*-
import torch


class PPORolloutBuffer:
    """
    简单时间步缓冲：按 {s, a, logp, v, r, s_next, done} 存
    你可以一轮存一条（server每轮一次），或累积多轮再更新
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.data = []

    def add(self, **kw):
        # 自动搬到device
        rec = {}
        for k, v in kw.items():
            if torch.is_tensor(v):
                rec[k] = v.detach().to(self.device)
            else:
                rec[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        self.data.append(rec)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data = []

    def stack(self):
        # 把 list[dict(tensor)] 堆成 dict(tensor[T,*])
        keys = self.data[0].keys()
        out = {}
        for k in keys:
            vs = [r[k] for r in self.data]
            if torch.is_tensor(vs[0]):
                out[k] = torch.stack(vs, dim=0)
            else:
                out[k] = torch.tensor(vs)
        return out