import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import read_client_data
from models.mnist_net import MNISTNet
from models.mini_vgg import MiniVGG
from PPO import TwoStageConfig, TwoStagePPO


class Server(object):
    def __init__(self, device, clients, dataset, model_name='MiniVGG',
                 prune_bins=(0., 0.1, 0.2, 0.3, 0.4), E_min=1, E_max=5, hidden=256, batch_norm=True,
                 warmup_rounds=10):
        self.device = device  # 'cpu' or 'cuda'
        self.clients = clients  # list of Client objects
        self.dataset = dataset  # string
        self.round_id = 0
        self.init_models_save_path = './init_models/'
        self.model_name = model_name
        self.server_model = MiniVGG(dataset=self.dataset, batch_norm=batch_norm)
        self.prev_acc = 0.0  # 上一轮全局模型精度

        # ==== Two-Stage PPO 初始化 ====
        self.ppo_config = TwoStageConfig(device=self.device)
        # 对齐两阶段 PPO 的输入维度（逐客户端 + 全局）
        self.ppo_config.s1.d_client = 6   # [Tn, acc_i, Dn, H_i, p_prev_i, En]
        self.ppo_config.s1.d_global = 6   # [Tmin, Tmean, Tmax, Tstd, acc_now, N]
        self.ppo_config.s2.d_client = 6   # [Tn, H_i, Dn, acc_i, p_i, En]
        self.ppo_config.s2.d_global = 6

        # 剪枝动作离散档
        self.ppo_config.s1.num_bins = len(prune_bins)
        self.ppo_config.s1.p_low = float(min(prune_bins))
        self.ppo_config.s1.p_high = float(max(prune_bins))
        self.ppo_config.s1.prune_bins = prune_bins
        # 轮数范围与总预算（softmax 后平均分配的期望）
        self.ppo_config.s2.E_min = E_min
        self.ppo_config.s2.E_max = E_max
        self.ppo_config.s2.tau_total = max(1, len(self.clients) * max(1, E_max) // 2)

        # 公共超参（可按需微调）
        self.ppo_config.common.lr = 3e-4
        self.ppo_config.common.clip_coef = 0.2
        self.ppo_config.common.gae_lambda = 0.95

        # 创建两阶段 PPO 管理器
        self.agent = TwoStagePPO(self.ppo_config)
        self.agent.cfg.s1.prune_bins = prune_bins
        # 兼容旧的规则策略读取路径
        self.agent.cfg.p_low = self.ppo_config.s1.p_low
        self.agent.cfg.p_high = self.ppo_config.s1.p_high
        self.agent.cfg.E_min = self.ppo_config.s2.E_min
        self.agent.cfg.E_max = self.ppo_config.s2.E_max

        self.ppo_update_every = 3

        # --- Warmup & Mixing ---
        self.warmup_rounds = warmup_rounds  # 前10轮不用PPO，只用规则/混合
        self.mix_epsilon_start = 1.0  # 早期完全规则
        self.mix_epsilon_end = 0.0  # 退火到纯PPO
        self.mix_epsilon_decay_rounds = 0  # ε 线性退火轮数

        # --- 动作稳定 / 冷却 / TTL ---
        self.delta_eps = getattr(self, "delta_eps", 0.05)  # 剪枝率变更的滞回阈值（5%）
        self.cooldown_rounds = getattr(self, "cooldown_rounds", 3)  # 至少间隔3轮才允许更换p
        self.client_last_change = {c.id: -10 for c in self.clients}

        # --- 奖励权重（可选：这里给默认值，方便从 self 读取）---
        self.lambda_p = getattr(self, "lambda_p", 0.5)  # 剪枝率变更惩罚
        self.alpha_acc = getattr(self, "alpha_acc", 1.0)  # 进展项权重
        self.beta_level = getattr(self, "beta_level", 0.4)  # 水平项权重
        self.kappa_eff = getattr(self, "kappa_eff", 0.3)  # 效率项权重
        self.lambda_T = getattr(self, "lambda_T", 0.3)  # 时间离散度惩罚
        self.lambda_E = getattr(self, "lambda_E", 0.0)  # （可选）算力感知 E 成本
        self.lambda_budget = getattr(self, "lambda_budget", 0.0)  # （可选）总预算
        self.E0_mean = getattr(self, "E0_mean", 10)  # （可选）期望平均本地轮数

        self.rewards = []
        self.client_wait_times = []
        self.total_run_time = 0.

        # ========= 这里添加：EMA 归一化器（只初始化一次） =========
        class _EmaNorm:
            def __init__(self, a=0.05):
                self.a = a
                self.m = 0.0
                self.v = 1.0

            @property
            def s(self):
                return max(self.v, 1e-6) ** 0.5

            def update(self, x: float):
                self.m += self.a * (x - self.m)
                self.v += self.a * ((x - self.m) ** 2 - self.v)

            def norm(self, x: float) -> float:
                return (x - self.m) / (self.s + 1e-6)

        self._EmaNorm = _EmaNorm
        self.norm_imp = _EmaNorm(0.05)  # 相对进展（Imp = acc_now - ema_acc）
        self.norm_tmax = _EmaNorm(0.05)  # 最大本地时间
        self.norm_stdT = _EmaNorm(0.05)  # 本地时间的标准差
        self.norm_dp = _EmaNorm(0.05)  # 平均 Δp
        self.norm_ci = _EmaNorm(0.05)  # （可选）每轮平均耗时 c_i = T_i / E_i，用于 E 成本

        # 进展/水平项需要的基线
        self.ema_acc = 0.0
        self.best_acc = 0.0
        # ======================================================

    def _init_server_model(self, batch_norm=True):
        if self.model_name == 'MiniVGG':
            return MiniVGG(dataset=self.dataset, batch_norm=batch_norm)
        if self.model_name == 'MnistNet':
            return MNISTNet()
        raise NotImplementedError

    def _rule_policy(self, times, entropies):
        N = len(times)
        T = np.array(times, dtype=float)
        H = np.array(entropies, dtype=float)

        # 归一化
        T_rank = (T.argsort().argsort() / max(N - 1, 1)) if N > 1 else np.zeros_like(T)
        H_norm = (H - H.min()) / max(H.max() - H.min(), 1e-6)

        p_low, p_high = self.agent.cfg.p_low, self.agent.cfg.p_high
        E_min, E_max = self.agent.cfg.E_min, self.agent.cfg.E_max
        p0 = 0.5 * (p_low + p_high)

        # ---- 剪枝率 P_i：慢 + 不确定 → 模型更大 ----
        p_rule = np.clip(p0 - 0.15 * T_rank - 0.10 * H_norm, p_low, p_high)

        # ---- 训练轮数 E_i：慢 → 少轮；不确定 → 多轮 ----
        E_rule = np.clip(np.round(
            E_min + (E_max - E_min) * (0.5 * H_norm - 0.25 * T_rank)
        ), E_min, E_max).astype(int)

        return p_rule.tolist(), E_rule.tolist()

    def _mix_actions(self, p_ppo, E_ppo, p_rule, E_rule):
        """
        ε-greedy 混合动作。热身期优先用规则；退火到纯PPO。
        """
        N = len(p_ppo)
        # 线性退火
        eps = max(self.mix_epsilon_end,
                  self.mix_epsilon_start - (self.round_id / max(1, self.mix_epsilon_decay_rounds)) *
                  (self.mix_epsilon_start - self.mix_epsilon_end))
        # 热身期强制规则
        use_rule_prob = eps if self.round_id > self.warmup_rounds else 1.0

        p_next, E_next = [], []
        for i in range(N):
            if np.random.rand() < use_rule_prob:
                p_next.append(float(p_rule[i]))
                E_next.append(int(E_rule[i]))
            else:
                p_next.append(float(p_ppo[i]))
                E_next.append(int(E_ppo[i]))
        return p_next, E_next

    def _allow_change(self, client_id, old_p, new_p):
        """冷却 + 滞回：若变化未达阈值或冷却未过，则不更换p"""
        if abs(new_p - old_p) < self.delta_eps:
            return False
        last = self.client_last_change.get(client_id, -10)
        if (self.round_id - last) < self.cooldown_rounds:
            return False
        self.client_last_change[client_id] = self.round_id
        return True

    def feddra_do(self):
        start_time = time.time()
        self.generate_next_round_params()  # 收集上一轮FL指标
        self.aggregate()  # 聚合模型，生成并且分发模型至client
        end_time = time.time()
        self.total_run_time += end_time - start_time

    def send_model_to_clients(self):
        """
        每个client接受聚合后的模型
        """
        state = {k: v.detach().clone() for k, v in self.server_model.state_dict().items()}
        for client in self.clients:
            client.model.load_state_dict(state)

    def generate_next_round_params(self):
        """
        Two-Stage PPO 流程（参考 HAPFL）：
          A. 收集本轮各 client 的结果（对应“上一轮动作”的效果）
          B. 构造两阶段状态：s1（逐客户端）与 s2_global（全局）
          C. 写入两阶段 (s,a,r) 到各自 PPO 缓冲
          D. 用 PPO1 产生“下一轮剪枝率”，用 PPO2 产生“下一轮训练轮数”
          E. 规则混合 / 冷却滞回，设置到各客户端
          F. 条件触发：分别更新两套 PPO
        约定 client.feddra_do() 返回：
          acc, total_time, entropy, local_data_size, client_id,
          client_last_pruning_rate, client_epochs, avg_loss, client_do_time
        """
        # ===== A) 收集指标 =====
        times, entropies, sizes, do_times = [], [], [], []
        p_exec, E_exec, losses, ids, accs, prev_p = [], [], [], [], [], []

        for client in self.clients:
            acc, total_time, entropy, local_data_size, client_id, \
                client_last_pruning_rate, client_epochs, avg_loss, client_do_time = client.feddra_do()

            times.append(float(total_time))
            entropies.append(float(entropy))
            sizes.append(int(local_data_size))
            p_exec.append(float(client_last_pruning_rate))
            E_exec.append(int(client_epochs))
            losses.append(float(avg_loss))
            ids.append(int(client_id))
            accs.append(float(acc))
            prev_p.append(float(client.last_pruning_rate))
            do_times.append(float(client_do_time))

        N = len(self.clients)
        if N == 0:
            return {}

        total_client_wait_time = self.cal_wait_time(do_times)
        self.client_wait_times.append(total_client_wait_time)

        # ===== B) 构造两阶段状态 =====
        # 全局精度 & 差分
        acc_now = float(sum(accs) / max(len(accs), 1))
        prev_acc = float(getattr(self, "prev_acc", 0.0))
        dAcc = acc_now - prev_acc

        # 单位 epoch 时间（上一轮实际执行结果）
        T_epoch = np.array([t / max(e, 1) for t, e in zip(times, E_exec)], dtype=np.float32)
        Tmin = float(T_epoch.min()) if len(T_epoch) else 1.0
        Tmean = float(T_epoch.mean()) if len(T_epoch) else 0.0
        Tmax = float(T_epoch.max()) if len(T_epoch) else 0.0
        Tstd = float(T_epoch.std()) if len(T_epoch) else 0.0

        # 逐客户端归一化特征
        D = np.array(sizes, dtype=np.float32)
        Dn = (D - (D.min() if len(D) else 0.0)) / (max(D.max() - (D.min() if len(D) else 0.0), 1e-6))
        En = np.array(E_exec, dtype=np.float32) / max(self.agent.cfg.E_max, 1)

        Tn = (T_epoch / max(Tmin, 1e-6)) if len(T_epoch) else np.zeros(N, dtype=np.float32)

        # —— Stage-1 输入（逐客户端）: [Tn, acc_i, Dn, H_i, p_prev, En]
        s1_rows = []
        for i in range(N):
            s1_rows.append([
                float(Tn[i]),
                float(accs[i]),
                float(Dn[i]),
                float(entropies[i]),
                float(prev_p[i]),
                float(En[i]),
            ])
        S1_cli = torch.tensor(s1_rows, dtype=torch.float32, device=self.device)  # [N, 6]

        # 全局上下文 g
        g = torch.tensor([Tmin, Tmean, Tmax, Tstd, acc_now, float(N)],
                         dtype=torch.float32, device=self.device)               # [6]

        print(f"[dbg] S1_cli shape: {S1_cli.shape}, g shape: {g.shape}")
        # 期望：S1_cli = [N, 6], g = [6]

        # ===== C) 两阶段奖励（基于上一轮结果） =====
        # —— Stage-1：逐客户端奖励向量（慢→剪多、难/数据多→剪少；附秩一致项）
        # 时间项
        r_time = - (torch.tensor(Tn, device=self.device) - 1.0) ** 2                        # [N]
        # 秩一致：rank(Tn) ≈ rank(p_exec)
        def _rank01(x_np):
            if len(x_np) <= 1: return np.zeros_like(x_np, dtype=np.float32)
            return (x_np.argsort().argsort().astype(np.float32) / max(len(x_np) - 1, 1))
        rank_t = torch.tensor(_rank01(Tn), device=self.device)
        rank_p = torch.tensor(_rank01(np.array(p_exec, dtype=np.float32)), device=self.device)
        r_align = 1.0 - torch.abs(rank_t - rank_p)                                            # [N], ∈[0,1]
        # 轻度精度项
        acc_t = torch.tensor(accs, dtype=torch.float32, device=self.device)
        if not hasattr(self, "ema_acc"): self.ema_acc = float(acc_now)
        ema = torch.tensor(self.ema_acc, device=self.device)
        r_acc = (acc_t - ema) / (acc_t.std() + 1e-6)

        w_time, w_align, w_acc = 1.0, 0.2, 0.1
        R1_vec = (w_time * r_time + w_align * r_align + w_acc * r_acc).unsqueeze(-1)          # [N,1]

        # —— Stage-2：全局奖励（越小越好）= - (Tmax - Tmin)
        R2 = float(Tmin - Tmax)

        print(f"[Server] Round {self.round_id} R1(mean)={R1_vec.mean().item():.4f}, R2={R2:.4f}")

        # ===== D) 写入 buffer（严格 on-policy：使用上一轮缓存的 Stage-1 决策） =====
        if hasattr(self, "_last_stage1"):
            last = self._last_stage1
            self.agent.store_transition_stage1(
                s=(last["S1"], last["g"]),
                a=last["a"],
                logp=last["logp"],
                v=last["v"],
                r=R1_vec.detach(),
                s_next=(S1_cli.detach(), g.detach()),
                done=torch.zeros(N, 1, device=self.device)
            )
            del self._last_stage1

        # Stage-2（critic-only）：使用全局上下文 g
        self.agent.store_transition_stage2(
            s=g.view(1, -1),
            logits=torch.zeros(1, N, device=self.device),
            v=torch.zeros(1, 1, device=self.device),
            r=torch.tensor([[R2]], dtype=torch.float32, device=self.device),
            s_next=g.view(1, -1),
            done=torch.zeros(1, 1, device=self.device)
        )

        # ===== E) 用策略为“下一轮”生成动作 =====
        with torch.no_grad():
            # PPO1：逐客户端剪枝率
            out1 = self.agent.select_pruning(S1_cli, g, deterministic=False)
            p_ppo = out1["p"].squeeze(-1).tolist()

            # 缓存“本轮下发的 Stage-1 决策”，下一轮用来写 buffer
            self._last_stage1 = dict(
                S1=S1_cli.detach().clone(),
                g=g.detach().clone(),
                a=out1["a"].detach().clone(),
                logp=out1["logp"].detach().clone(),
                v=out1["v"].detach().clone(),
            )

            # 构造 Stage-2 逐客户端特征: [Tn, H_i, Dn, acc_i, p_i(next), En]
            S2_rows = []
            p_next_vec = out1["p"].squeeze(-1).clamp(0.0, 1.0).cpu().numpy()
            for i in range(N):
                S2_rows.append([
                    float(Tn[i]),
                    float(entropies[i]),
                    float(Dn[i]),
                    float(accs[i]),
                    float(p_next_vec[i]),
                    float(En[i]),
                ])
            S2_cli = torch.tensor(S2_rows, dtype=torch.float32, device=self.device)  # [N,6]

            out2 = self.agent.select_epochs(
                S2_cli, g,
                tau_total=int(self.agent.cfg.s2.tau_total),
                E_min=int(self.agent.cfg.E_min),
                E_max=int(self.agent.cfg.E_max),
                deterministic=False
            )
            E_ppo = [int(x) for x in out2["E"].tolist()]

        # 规则策略（便于 warmup 与混合）
        p_rule, E_rule = self._rule_policy(times, entropies)
        p_next, E_next = self._mix_actions(p_ppo, E_ppo, p_rule, E_rule)

        # —— 冷却/滞回并下发（附：吸附到最近bin + E边界裁剪）——
        use_bins = getattr(self.agent.cfg.s1, "use_discrete_bins", True)
        bins = None
        if use_bins:
            # 注意：bins 必须与 PPO1 的 prune_bins 一致
            bins = torch.tensor(self.agent.cfg.s1.prune_bins, dtype=torch.float32).cpu()

        E_min = int(getattr(self.agent.cfg, "E_min", 1))
        E_max = int(getattr(self.agent.cfg, "E_max", 19))

        def snap_to_bin(x: float) -> float:
            if bins is None:  # 连续策略时不吸附
                return float(x)
            idx = int(torch.argmin(torch.abs(bins - x)).item())
            return float(bins[idx].item())

        for i, client in enumerate(self.clients):
            old_p = float(getattr(client, "cur_pruning_rate", p_next[i]))
            # 先对 p_next 做一次吸附（保证规则混合/数值扰动后也落在 bins 上）
            cand_p = snap_to_bin(float(p_next[i])) if use_bins else float(p_next[i])

            # 冷却/滞回：若变化小于阈值或未过冷却期，则保持原值
            new_p = cand_p
            final_p = new_p if self._allow_change(client.id, old_p, new_p) else old_p

            # 连续策略下再做边界裁剪；离散策略不需要 clip(0,1)
            if not use_bins:
                final_p = float(np.clip(final_p, float(self.agent.cfg.p_low), float(self.agent.cfg.p_high)))

            # E 做硬边界（PPO2 的 softmax 分配可能给到 0 或过大）
            Ei = int(np.clip(int(E_next[i]), E_min, E_max))

            client.cur_pruning_rate = final_p
            client.training_intensity = Ei

            print(f"[Round {self.round_id}] Client {client.id} -> next p={client.cur_pruning_rate:.4f}, E={client.training_intensity}")

        # ===== F) 统计刷新 & 条件更新 PPO =====
        self.prev_acc = acc_now
        self.round_id += 1

        # 触发两套 PPO 更新（可沿用原来的 ppo_update_every）
        if (self.round_id > self.warmup_rounds) and (
                len(self.agent.buf1) + len(self.agent.buf2) >= self.ppo_update_every):
            out_s1 = self.agent.ppo_update_stage1()
            out_s2 = self.agent.ppo_update_stage2()
            print("PPO1 updated ✅", out_s1)
            print("PPO2 updated ✅", out_s2)

        # 返回（可选）
        return {c.id: dict(pruning_rate=float(getattr(c, "cur_pruning_rate", 0.0)),
                           epochs=int(getattr(c, "training_intensity", 1)))
                for c in self.clients}

    def aggregate(self):
        server_model = self.server_model
        for param in server_model.parameters():
            param.data.zero_()  # 将簇模型参数都设置为0
        client_models = []
        for client in self.clients:
            client_model = client.load_model()
            client_models.append(client_model)
            client_mask = client_model.mask
            ratio = 1. / len(self.clients)  # ratio 为 1 / n
            layer_idx_in_mask = 0
            if self.dataset == 'MNIST' or self.dataset == 'emnist_noniid':
                start_mask_client = torch.ones(1).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            else:
                start_mask_client = torch.ones(3).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            end_mask_client = torch.tensor(client_mask[layer_idx_in_mask], dtype=torch.int).bool()
            client_model = client_model.to(self.device)
            server_model = server_model.to(self.device)
            start_mask_client = start_mask_client.to(self.device)
            end_mask_client = end_mask_client.to(self.device)
            for client_layer, server_layer in zip(client_model.modules(), server_model.modules()):
                start_indices = [i for i, x in enumerate(start_mask_client) if x]
                end_indices = [i for i, x in enumerate(end_mask_client) if x]
                if isinstance(client_layer, nn.BatchNorm2d):
                    # with torch.no_grad():
                    #     cluster_layer.weight.data[end_indices] += ratio * client_layer.weight.data
                    #     cluster_layer.bias.data[end_indices] += ratio * client_layer.bias.data
                    # cluster_layer.running_mean.data[end_indices] += ratio * client_layer.running_mean.data
                    # cluster_layer.running_var.data[end_indices] += ratio * client_layer.running_var.data
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Conv2d):
                    with torch.no_grad():
                        for i, start_idx in enumerate(start_indices):
                            server_layer.weight.data[end_indices, start_idx, :, :] += ratio * client_layer.weight.data[
                                                                                              :, i, :, :]

                if isinstance(client_layer, nn.Linear):
                    with torch.no_grad():
                        for i, start_idx in enumerate(start_indices):
                            server_layer.weight.data[end_indices, start_idx] += ratio * client_layer.weight.data[:, i]
                        server_layer.bias.data[end_indices] += ratio * client_layer.bias.data
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
        # 此时cluster_model已完成异构模型聚合
        # 每个client根据自己的模型结构，从cluster_model中获取子模型
        self.server_model = server_model
        for i, client in enumerate(self.clients):
            client_model = client_models[i]
            client_mask = client_model.mask
            layer_idx_in_mask = 0
            if self.dataset == 'MNIST' or self.dataset == 'emnist_noniid':
                start_mask_client = torch.ones(1).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            else:
                start_mask_client = torch.ones(3).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            end_mask_client = torch.tensor(client_mask[layer_idx_in_mask], dtype=torch.int).bool()
            for client_layer, server_layer in zip(client_model.modules(), server_model.modules()):
                start_indices = [i for i, x in enumerate(start_mask_client) if x]
                end_indices = [i for i, x in enumerate(end_mask_client) if x]
                if isinstance(client_layer, nn.BatchNorm2d):
                    # with torch.no_grad():
                    #     client_layer.weight.data = cluster_layer.weight.data[end_indices].clone()
                    #     client_layer.bias.data = cluster_layer.bias.data[end_indices].clone()
                    # client_layer.running_mean.data = cluster_layer.running_mean.data[end_indices].clone()
                    # client_layer.running_var.data = cluster_layer.running_var.data[end_indices].clone()
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Linear):
                    m0 = server_layer.weight.data[end_indices, :].clone()
                    with torch.no_grad():
                        client_layer.weight.data = m0[:, start_indices].clone()
                        client_layer.bias.data = server_layer.bias.data[end_indices].clone()
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Conv2d):
                    m0 = server_layer.weight.data[end_indices, :, :, :].clone()
                    with torch.no_grad():
                        client_layer.weight.data = m0[:, start_indices].clone()
            # 存储client_model
            client.model = client_model

    def fedavg_do(self):
        start_time = time.time()
        client_do_time = []
        for client in self.clients:
            do_time = client.fedavg_do()
            client_do_time.append(do_time)

        wait_time = self.cal_wait_time(client_do_time)
        self.client_wait_times.append(wait_time)
        self.fedavg_aggregate()
        state = {k: v.detach().clone() for k, v in self.server_model.state_dict().items()}
        for client in self.clients:
            client.model.load_state_dict(state)
        end_time = time.time()
        self.total_run_time += end_time - start_time

    def fedavg_aggregate(self):
        server_model = self.server_model
        server_model = server_model.to(self.device)
        for param in server_model.parameters():
            param.data.zero_()
        client_num = len(self.clients)
        ratio = 1. / client_num
        for client in self.clients:
            client_model = client.model
            client_model = client_model.to(self.device)
            for server_param, client_param in zip(server_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * ratio
        self.server_model = server_model

    def fedbn_do(self):
        # fedbn方法，聚合除bn层以外参数
        start_time = time.time()
        client_do_time = []
        for client in self.clients:
            do_time = client.fedavg_do()
            client_do_time.append(do_time)
        wait_time = self.cal_wait_time(client_do_time)
        self.client_wait_times.append(wait_time)
        self.aggregate()
        end_time = time.time()
        self.total_run_time += end_time - start_time

    def fedbn_aggregate(self):
        pass


    def cal_wait_time(self, total_time):
        # 根据client使用的total_time计算客户端等待的时间
        max_time = max(total_time)
        wait_time = sum(max_time - t for t in total_time)
        return wait_time
