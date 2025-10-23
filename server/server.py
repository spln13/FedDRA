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
from PPO import DualStreamConfig, DualStreamPPO


class Server(object):
    def __init__(self, device, clients, dataset, model_name='MiniVGG',
                 d_glob=6, d_cli=7, p_low=0.2, p_high=0.9, E_min=1, E_max=5, hidden=256, batch_norm=True,
                 warmup_rounds=10):
        self.device = device  # 'cpu' or 'cuda'
        self.clients = clients  # list of Client objects
        self.dataset = dataset  # string
        self.round_id = 0
        self.prev_acc = 0.0  # 上一轮全局模型精度
        self.ppo_config = DualStreamConfig(d_glob=d_glob, d_cli=d_cli,
                                           p_low=p_low, p_high=p_high, E_min=E_min, E_max=E_max,
                                           hidden=hidden)

        self.agent = DualStreamPPO(self.ppo_config)
        self.ppo_update_every = 3
        self.init_models_save_path = './init_models/'
        self.model_name = model_name
        self.server_model = MiniVGG(dataset=self.dataset, batch_norm=batch_norm)

        # --- Warmup & Mixing ---
        self.warmup_rounds = warmup_rounds  # 前10轮不用PPO，只用规则/混合
        self.mix_epsilon_start = 1.0  # 早期完全规则
        self.mix_epsilon_end = 0.0  # 退火到纯PPO
        self.mix_epsilon_decay_rounds = 20  # ε 线性退火轮数

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
        Server 流程：
          A. 收集本轮各 client 的结果指标（这些指标对应“本轮动作”的效果）
          B. 构造 PPO 的 S_glob / S_cli / mask
          C. 写入 (state, action, reward) 到 PPO buffer
          D. 用策略生成“下一轮动作”并下发
          E. 到达间隔则 ppo_update()
        约定 client.feddra_do() 返回：
          acc, total_time, entropy, local_data_size, client_id,
          client_last_pruning_rate, client_epochs, avg_loss, client_do_time
        """

        # ===== A) 收集 client 指标（本轮已执行的动作及结果） =====
        times, entropies, sizes, do_times = [], [], [], []
        p_exec, E_exec, losses, ids, accs, prev_p = [], [], [], [], [], []

        for client in self.clients:
            acc, total_time, entropy, local_data_size, client_id, \
                client_last_pruning_rate, client_epochs, avg_loss, client_do_time = client.feddra_do()

            times.append(float(total_time))
            entropies.append(float(entropy))
            sizes.append(int(local_data_size))
            p_exec.append(float(client_last_pruning_rate))  # 本轮“实际执行”的剪枝率
            E_exec.append(int(client_epochs))  # 本轮“实际执行”的本地轮数
            losses.append(float(avg_loss))
            ids.append(int(client_id))
            accs.append(float(acc))
            prev_p.append(float(client.last_pruning_rate))  # 上轮下发/生效的剪枝率
            do_times.append(float(client_do_time))

        N = len(self.clients)
        if N == 0:
            return {}

        total_client_wait_time = self.cal_wait_time(do_times)
        print(f"[Round {self.round_id}] Total client wait time: {total_client_wait_time:.2f}s")

        # ===== B) 构造 PPO 状态 =====
        # 全局特征 S_glob: [Acc, dAcc, Tmax, dT, round_id, bw]
        acc_now = float(sum(accs) / max(len(accs), 1))
        prev_acc = float(getattr(self, "prev_acc", 0.0))
        dAcc = acc_now - prev_acc
        Tmax = float(max(times)) if times else 0.0
        dT = float(max(times) - min(times)) if len(times) > 1 else 0.0  # 继续保留给 S_glob
        stdT = float(np.std(np.array(times, dtype=np.float32))) if len(times) > 1 else 0.0  # 用于奖励更稳
        bw = 1.0
        # Δp：逐客户端平均绝对差
        dP = float(
            np.mean(np.abs(np.array(p_exec, dtype=np.float32) - np.array(prev_p, dtype=np.float32)))
        ) if N > 0 else 0.0

        S_glob = torch.tensor([[acc_now, dAcc, Tmax, dT, float(getattr(self, "round_id", 0)), bw]],
                              dtype=torch.float32, device=self.device)  # [1, 6]

        # 客户端特征 S_cli（7维）：[T_i, H_i, |D_i|, id_emb, p_prev, E_prev, loss_prev]
        rows = []
        for i in range(N):
            rows.append([
                times[i],
                entropies[i],
                sizes[i],
                float(ids[i]),  # 可换成 learnable embedding
                p_exec[i],  # 本轮实际动作：剪枝率
                E_exec[i],  # 本轮实际动作：epochs
                losses[i],
            ])
        S_cli = torch.tensor([rows], dtype=torch.float32, device=self.device)  # [1, N, 7]
        mask = torch.ones(1, N, dtype=torch.float32, device=self.device)
        prev_p_tensor = torch.tensor([[[x] for x in prev_p]], dtype=torch.float32, device=self.device)

        # ===== C) 写入经验 (state, action, reward) =====
        # ---- 奖励：进展(Imp) + 水平(Level) + 效率(Eff) - 时间异质性(stdT) - Δp + 软SLA (+可选E成本/预算) ----
        # 首轮/基线保护
        if self.round_id == 0 and getattr(self, "ema_acc", 0.0) == 0.0:
            self.ema_acc = acc_now
            self.best_acc = acc_now

        # 进展：相对 EMA 基线（避免后期 dAcc→0 无奖励）
        Imp_raw = acc_now - self.ema_acc

        # 水平：相对目标（历史最优的 95% 或不低于 0.8）
        A_target = max(0.8, 0.95 * float(getattr(self, "best_acc", acc_now)))
        tau_A = 0.02
        Level_raw = 1.0 / (1.0 + np.exp(-(acc_now - A_target) / max(tau_A, 1e-6)))  # sigmoid(acc_now - A_target)

        # 使用 __init__ 里创建的 EMA 归一化器
        Imp_n = self.norm_imp.norm(Imp_raw)
        Tmax_n = self.norm_tmax.norm(Tmax)
        stdT_n = self.norm_stdT.norm(stdT)
        dP_n = self.norm_dp.norm(dP)

        # 效率：进展/时间（归一化尺度内计算）
        Eff = Imp_n / (abs(Tmax_n) + 1e-3)

        # 软 SLA：Tmax 超标时柔性惩罚
        T_target = float(np.median(np.array(times, dtype=np.float32)) * 1.1) if len(times) else 0.0
        tau_T = 0.5
        r_sla = - float(torch.nn.functional.softplus(
            torch.tensor((Tmax - T_target) / (tau_T + 1e-6))
        ).item())

        # 合成奖励（从 self 读取系数）
        alpha = float(getattr(self, "alpha_acc", 1.0))  # 进展
        beta = float(getattr(self, "beta_level", 0.4))  # 水平
        kappa = float(getattr(self, "kappa_eff", 0.3))  # 效率
        lam_T = float(getattr(self, "lambda_T", 0.3))  # 异质性
        lam_p = float(getattr(self, "lambda_p", 0.2))  # Δp 平滑

        R = alpha * Imp_n + beta * Level_raw + kappa * Eff \
            - lam_T * stdT_n - lam_p * dP_n + r_sla

        # （可选）E 成本：算力/速度感知；只有当 self.lambda_E>0 才启用
        lambda_E = float(getattr(self, "lambda_E", 0.0))
        if lambda_E > 0.0 and N > 0:
            ci_list = []
            for i in range(N):
                ci = times[i] / max(E_exec[i], 1)  # 每轮平均耗时（慢设备更大）
                self.norm_ci.update(ci)
                ci_list.append(self.norm_ci.norm(ci))
            cost_E = float(np.mean([ci_list[i] * E_exec[i] for i in range(N)]))
            R -= lambda_E * cost_E

        # （可选）总预算：sum(E_i) 不宜超过 N*E0；只有当 self.lambda_budget>0 才启用
        lambda_budget = float(getattr(self, "lambda_budget", 0.0))
        if lambda_budget > 0.0 and N > 0:
            E0 = int(getattr(self, "E0_mean", 10))  # 期望均值（例如 10）
            B = N * E0
            over = max(0, int(sum(E_exec)) - B)
            R -= lambda_budget * (over ** 2)

        print(
            f"Round {self.round_id}: acc_now={acc_now:.4f} dAcc={dAcc:.4f} Tmax={Tmax:.2f} stdT={stdT:.2f} dP={dP:.4f} -> Reward={R:.4f}")
        self.rewards.append(float(R))

        # 更新 EMA 基线与统计
        self.ema_acc = 0.95 * self.ema_acc + 0.05 * acc_now
        self.best_acc = max(self.best_acc, acc_now)
        self.norm_imp.update(Imp_raw)
        self.norm_tmax.update(Tmax)
        self.norm_stdT.update(stdT)
        self.norm_dp.update(dP)

        # 2) 动作张量（形状与 PPO 网络一致）
        p_tensor = torch.tensor([[[x] for x in p_exec]], dtype=torch.float32, device=self.device)  # [1, N, 1]
        E_tensor = torch.tensor([[[x] for x in E_exec]], dtype=torch.float32, device=self.device)  # [1, N, 1]

        # 3) 价值与 logp
        #    ✅ 最佳：用“下发时缓存的 old_logp”；若无，这里 evaluate 近似
        eval_out = self.agent.net.evaluate_actions(S_glob, S_cli, p_tensor, E_tensor, mask)
        V_now = eval_out["V"].detach()  # [1, N, 1]
        logp_now = eval_out["logp"].detach()  # [1, N]

        # 4) 写入 buffer
        self.agent.store_transition(dict(
            S_glob=S_glob, S_cli=S_cli, mask=mask,
            p=p_tensor, E=E_tensor,
            logp=logp_now,  # 若你有缓存 old_logp，请替换此处
            V=V_now,
            reward=torch.tensor([[R]], dtype=torch.float32, device=self.device),
            done=torch.zeros(1, 1, device=self.device),
            prev_p=prev_p_tensor
        ))

        # ===== D) 用策略为“下一轮”生成动作并下发 =====
        with torch.no_grad():
            out_next = self.agent.select_actions(S_glob, S_cli, mask, prev_p=prev_p_tensor)
        p_ppo = out_next["p"][0, :, 0].tolist()
        E_ppo = [int(x) for x in out_next["E"][0, :, 0].tolist()]

        # 规则动作
        p_rule, E_rule = self._rule_policy(times, entropies)
        # ε-混合（热身期优先规则，随后退火）
        p_next, E_next = self._mix_actions(p_ppo, E_ppo, p_rule, E_rule)

        # 冷却/滞回：是否采纳新 p；并设置下一轮的训练轮数
        for i, client in enumerate(self.clients):
            old_p = float(getattr(client, "cur_pruning_rate", p_next[i]))
            new_p = float(p_next[i])
            final_p = new_p if self._allow_change(client.id, old_p, new_p) else old_p
            client.cur_pruning_rate = final_p
            client.training_intensity = int(E_next[i])
            # 可选：缓存下发时的 logp，严格 on-policy 时在下轮写入 buffer
            client.last_action_logp = float(out_next["logp"][0, i].item())
            print(f"[Round {self.round_id}] Client {client.id} -> next p={final_p:.4f}, E={client.training_intensity}")

        # ===== E) 更新 PPO =====
        self.prev_acc = acc_now
        self.round_id += 1

        if (self.round_id > self.warmup_rounds) and (len(self.agent.buffer.data) >= self.ppo_update_every):
            self.agent.ppo_update(epochs=4, batch_size=256, normalize_adv=True)
            print("PPO updated ✅")

        # 返回给上层（可选）
        return {c.id: dict(pruning_rate=float(getattr(c, "cur_pruning_rate", 0.0)),
                           epochs=int(getattr(c, "training_intensity", getattr(self.agent.cfg, "E_min", 1))))
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
