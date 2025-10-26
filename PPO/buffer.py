# PPO/buffer.py
from typing import Dict, List, Any, Tuple
import torch


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return tuple(_to_tensor(xx, device) for xx in x)
    return torch.tensor(x, dtype=torch.float32, device=device)


class SimpleTrajBuffer:
    """
    存 (s, a, logp, v, r, s_next, done, [extras...]) 的轻量 buffer。
    允许 s/s_next 是 (S_cli, g) 的二元组。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.data: List[Dict[str, Any]] = []

    def add(self, **kwargs):
        self.data.append({k: v for k, v in kwargs.items()})

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def stack(self) -> Dict[str, Any]:
        # 把列表里的字段逐个拼起来（按第0维）；
        # s / s_next 若为 tuple 则分别拼。
        out: Dict[str, Any] = {}
        if not self.data:
            return out

        keys = self.data[0].keys()
        for k in keys:
            vals = [d[k] for d in self.data]
            if k in ("s", "s_next") and isinstance(vals[0], tuple):
                # 二元组 (S_cli, g)
                S_cli = torch.cat([_to_tensor(v[0], self.device) for v in vals], dim=0)
                g = torch.cat([_to_tensor(v[1], self.device) for v in vals], dim=0)
                out[k] = (S_cli, g)
            else:
                # 普通张量（需要保证相同形状或能按第0维拼接）
                tens = []
                for v in vals:
                    v = _to_tensor(v, self.device)
                    tens.append(v if v.dim() > 0 else v.view(1))
                out[k] = torch.cat(tens, dim=0)
        return out
