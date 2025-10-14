import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ===================== 工具与通用部分 =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_raw_dataset(dataset_name: str, train: bool, root: str = "./data"):
    """返回 (images_uint8, labels_int64, num_classes)；不做标准化/ToTensor。"""
    name = dataset_name.lower()
    if name == "mnist":
        ds = datasets.MNIST(root, train=train, download=True, transform=None)
        # MNIST: data为 [N, 28, 28] 的uint8 tensor；targets为长整型
        imgs = ds.data.numpy()  # uint8, (N, 28, 28)
        labels = np.array(ds.targets, dtype=np.int64)
        num_classes = 10
        return imgs, labels, num_classes
    elif name in ("cifar10", "cifar-10"):
        ds = datasets.CIFAR10(root, train=train, download=True, transform=None)
        # CIFAR10: data为 [N, 32, 32, 3] 的uint8 ndarray；targets是list
        imgs = ds.data.astype(np.uint8)  # (N, 32, 32, 3)
        labels = np.array(ds.targets, dtype=np.int64)
        num_classes = 10
        return imgs, labels, num_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def indices_by_class(labels: np.ndarray, num_classes: int) -> List[List[int]]:
    buckets = [[] for _ in range(num_classes)]
    for i, y in enumerate(labels):
        buckets[y].append(i)
    for k in range(num_classes):
        random.shuffle(buckets[k])
    return buckets


def dirichlet_matrix(num_clients: int, num_classes: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    # 每行为一个客户端的类别比例，行和=1
    return rng.dirichlet(alpha * np.ones(num_classes), size=num_clients)


def split_by_weight(idx_per_class: List[List[int]], weights: np.ndarray) -> List[List[int]]:
    """把每个类别的索引按权重分配到各客户端；weights shape=(N,K)"""
    N, K = weights.shape
    out = [[] for _ in range(N)]
    for k in range(K):
        idx_k = idx_per_class[k]
        n_k = len(idx_k)
        if n_k == 0:
            continue
        w = weights[:, k]
        if np.allclose(w.sum(), 0.0):
            w = np.ones(N) / N
        else:
            w = w / w.sum()
        counts = np.random.multinomial(n_k, w)
        start = 0
        for j in range(N):
            c = counts[j]
            if c > 0:
                out[j].extend(idx_k[start:start + c])
                start += c
        # 若有余量（极小概率），全给最大权重客户端
        if start < n_k:
            rem = idx_k[start:]
            j_star = int(np.argmax(w))
            out[j_star].extend(rem)
    for j in range(N):
        random.shuffle(out[j])
    return out


def stratified_take(indices: List[int], labels: np.ndarray, take_per_class: Dict[int, int]) -> List[int]:
    by_cls = defaultdict(list)
    for i in indices:
        by_cls[int(labels[i])].append(i)
    for k in by_cls:
        random.shuffle(by_cls[k])
    chosen = []
    for k, need in take_per_class.items():
        bucket = by_cls.get(k, [])
        if need >= len(bucket):
            chosen.extend(bucket)
        else:
            chosen.extend(bucket[:need])
    random.shuffle(chosen)
    return chosen


# ===================== 保存为 .npz =====================

def save_client_npz_split(
        base_dir: str,
        dataset_name: str,
        split_name: str,  # "train" | "test" | "local_test"
        client_id: int,
        images: np.ndarray,
        labels: np.ndarray
):
    """将某客户端某split的数据存到 dataset_name/split/client_id.npz"""
    ddir = os.path.join(base_dir, dataset_name.lower(), split_name)
    ensure_dir(ddir)
    path = os.path.join(ddir, f"{client_id}.npz")
    # 统一用 uint8 存图，int64 存标签
    np.savez_compressed(path, images=images.astype(np.uint8), labels=labels.astype(np.int64))


# ===================== 主流程：生成并落盘 =====================

def build_and_dump(
        dataset_name: str,
        num_clients: int,
        alpha: float = 0.5,
        small_local_test_size: int = 200,
        seed: int = 42,
        base_dir: str = "./federated_data",
        download_root: str = "./data"
) -> Dict:
    """
    构建联邦划分并把每个客户端的 train/test/local_test 写成 .npz。
    返回 meta 信息（可记录在 json）。
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # 读原始 raw 数据（uint8）
    train_imgs, train_labels, K = get_raw_dataset(dataset_name, train=True, root=download_root)
    test_imgs, test_labels, _ = get_raw_dataset(dataset_name, train=False, root=download_root)

    train_idx_by_cls = indices_by_class(train_labels, K)
    test_idx_by_cls = indices_by_class(test_labels, K)

    # Dirichlet 类别分布矩阵（同一矩阵用于 train/test/local_test）
    P = dirichlet_matrix(num_clients, K, alpha, rng)  # (N, K)

    # 生成索引划分
    client_train_idx = split_by_weight(train_idx_by_cls, P)
    client_test_idx = split_by_weight(test_idx_by_cls, P)

    # meta / splits 记录（便于复现实验）
    meta = {
        "dataset": dataset_name,
        "num_clients": num_clients,
        "alpha": alpha,
        "small_local_test_size": small_local_test_size,
        "seed": seed,
        "note": "train/test/local_test 使用同一 Dirichlet 行向量保证同分布"
    }
    splits = {}

    # 为每个客户端写入 .npz
    for cid in range(num_clients):
        # ----- train -----
        tr_idx = client_train_idx[cid]
        tr_imgs = train_imgs[tr_idx]
        tr_lbls = train_labels[tr_idx]
        save_client_npz_split(base_dir, dataset_name, "train", cid, tr_imgs, tr_lbls)

        # ----- test -----
        te_idx = client_test_idx[cid]
        te_imgs = test_imgs[te_idx]
        te_lbls = test_labels[te_idx]
        save_client_npz_split(base_dir, dataset_name, "test", cid, te_imgs, te_lbls)

        # ----- local_test（从该客户端 test 中分层抽样，按 P[cid] 比例）-----
        probs = P[cid] / P[cid].sum()
        raw = probs * small_local_test_size
        per_class = np.floor(raw).astype(int)
        remain = small_local_test_size - per_class.sum()
        if remain > 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for t in range(remain):
                per_class[order[t]] += 1
        take_map = {int(k): int(per_class[k]) for k in range(K)}

        # 若客户端 test 太少，用“取尽”策略维持可用性
        if len(te_idx) <= small_local_test_size // 4:
            # 退化：直接把全部 test 作为 local_test
            lt_idx = te_idx
        else:
            lt_idx = stratified_take(te_idx, test_labels, take_map)

        lt_imgs = test_imgs[lt_idx] if len(lt_idx) > 0 else test_imgs[te_idx]
        lt_lbls = test_labels[lt_idx] if len(lt_idx) > 0 else test_labels[te_idx]
        save_client_npz_split(base_dir, dataset_name, "local_test", cid, lt_imgs, lt_lbls)

        # 记录
        splits[str(cid)] = {
            "train": len(tr_idx),
            "test": len(te_idx),
            "local_test": len(lt_idx)
        }

    # 保存 meta & 概览
    out_dir = os.path.join(base_dir, dataset_name.lower())
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "split_sizes.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    return {"meta": meta, "sizes": splits}


# ===================== 读取：供 PyTorch 训练 =====================

class NpzArrayDataset(Dataset):
    """
    从 .npz 读取到的 (images:uint8, labels:int64)，在 __getitem__ 中做 transforms。
    - MNIST: (H,W) -> [1,H,W] float，标准化为 0-1 后再 Normalize
    - CIFAR: (H,W,C) -> [C,H,W] float
    """

    def __init__(self, npz_path: str, dataset_name: str, train: bool):
        arr = np.load(npz_path)
        self.images = arr["images"]  # uint8
        self.labels = arr["labels"].astype(np.int64)
        self.dataset_name = dataset_name.lower()
        self.train = train

        if self.dataset_name == "mnist":
            self.mean = (0.1307,)
            self.std = (0.3081,)
        elif self.dataset_name in ("cifar10", "cifar-10"):
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
        else:
            raise ValueError("Unsupported dataset")

    def __len__(self):
        return self.labels.shape[0]

    def _to_tensor_and_norm(self, img_uint8: np.ndarray) -> torch.Tensor:
        if self.dataset_name == "mnist":
            # (H,W) -> [1,H,W]
            t = torch.from_numpy(img_uint8).unsqueeze(0).float() / 255.0
        else:
            # (H,W,C) -> [C,H,W]
            t = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
        # Normalize
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        if t.shape[0] != mean.shape[0]:
            # MNIST 情况下 t.shape[0]==1，mean/std 也为1；否则广播
            mean = mean[:t.shape[0]]
            std = std[:t.shape[0]]
        return (t - mean) / std

    def __getitem__(self, idx: int):
        x = self._to_tensor_and_norm(self.images[idx])
        y = int(self.labels[idx])
        return x, y


class Client:
    """
    表示一个客户端，提供读取 train/test/local_test 的 DataLoader。
    目录结构：
      base_dir/
        cifar10/
          train/0.npz ... N-1.npz
          test/0.npz  ... N-1.npz
          local_test/0.npz ... N-1.npz
        mnist/
          ...
    """

    def __init__(self, dataset_name: str, client_id: int, base_dir: str = "./federated_data"):
        self.dataset_name = dataset_name.lower()
        self.client_id = int(client_id)
        self.base_dir = base_dir

    def _npz_path(self, split: str) -> str:
        return os.path.join(self.base_dir, self.dataset_name, split, f"{self.client_id}.npz")

    def load_train_data(self, batch_size: int = 64, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("train"), self.dataset_name, train=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def load_test_data(self, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("test"), self.dataset_name, train=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def load_local_test_data(self, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("local_test"), self.dataset_name, train=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# ===================== 示例 =====================

if __name__ == "__main__":
    """
    运行示例：
    1) 生成并落盘 CIFAR-10 和 MNIST（各 N=5, alpha=0.3, local_test=200）
    2) 用 Client(0) 读回 DataLoader 并打印 batch 形状
    """
    set_seed(2025)
    BASE = "."

    # ------- 生成 CIFAR-10 -------
    build_and_dump(
        dataset_name="cifar10",
        num_clients=10,
        alpha=0.3,
        small_local_test_size=200,
        seed=2025,
        base_dir=BASE,
        download_root="./data"
    )

    # ------- 生成 MNIST -------
    build_and_dump(
        dataset_name="mnist",
        num_clients=10,
        alpha=0.5,
        small_local_test_size=100,
        seed=2025,
        base_dir=BASE,
        download_root="./data"
    )

    # ------- 读取并检查 -------
    c = Client(dataset_name="cifar10", client_id=0, base_dir=BASE)
    dl_tr = c.load_train_data(batch_size=128)
    dl_te = c.load_test_data(batch_size=128)
    dl_lt = c.load_local_test_data(batch_size=128)

    x, y = next(iter(dl_tr))
    print("[CIFAR10] Client0 train batch:", x.shape, y.shape)
    x, y = next(iter(dl_te))
    print("[CIFAR10] Client0 test  batch:", x.shape, y.shape)
    x, y = next(iter(dl_lt))
    print("[CIFAR10] Client0 local batch:", x.shape, y.shape)

    m = Client(dataset_name="mnist", client_id=0, base_dir=BASE)
    x, y = next(iter(m.load_train_data(batch_size=256)))
    print("[MNIST]   Client0 train batch:", x.shape, y.shape)
