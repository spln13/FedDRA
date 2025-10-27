# PPO/hapfl_buffer.py
from typing import Dict, List
import torch

class TrajBuf:
    def __init__(self, device: str = "cpu"):
        self.data: List[Dict[str, torch.Tensor]] = []
        self.device = torch.device(device)

    def add(self, **kw):
        rec = {k: (v.to(self.device) if torch.is_tensor(v) else torch.tensor(v, device=self.device))
               for k, v in kw.items()}
        self.data.append(rec)

    def __len__(self): return len(self.data)

    def clear(self): self.data.clear()

    def stack(self):
        out = {}
        for k in self.data[0].keys():
            vs = [rec[k] for rec in self.data]
            out[k] = torch.stack(vs, dim=0)
        return out
