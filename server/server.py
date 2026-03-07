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
        # 状态维度：
        #   s1: [t_norm, entropy_norm, size_norm, p_exec]
        #   s2: [tm_norm, entropy_norm, size_norm, p_next]
        self.ppo_config.s1.s1_dim = 4
        self.ppo_config.s2.s2_dim = 4
        self.ppo_config.s2.use_clientwise = True

        # 剪枝动作：离散档
        self.ppo_config.s1.use_discrete_bins = True
        self.ppo_config.s1.prune_bins = prune_bins

        # 轮数总预算（PPO2 会在 [E_min, E_max] 下做预算分配）
        self.ppo_config.s2.tau_total = max(
            len(self.clients) * max(1, E_min),
            len(self.clients) * max(1, E_max) // 2,
        )
        # 公共超参（可按需微调）
        self.ppo_config.common.lr = 3e-4
        self.ppo_config.common.clip_coef = 0.2
        self.ppo_config.common.gae_lambda = 0.95
        self.ppo_config.common.target_kl = 0.02

        # 创建两阶段 PPO 管理器
        self.agent = TwoStagePPO(self.ppo_config)
        # 兼容规则策略/其它函数对区间与边界的读取
        self.agent.cfg.p_low = min(self.ppo_config.s1.prune_bins)
        self.agent.cfg.p_high = max(self.ppo_config.s1.prune_bins)
        self.agent.cfg.E_min = E_min
        self.agent.cfg.E_max = E_max

        self.ppo_update_every = 4

        # --- Warmup & Mixing ---
        self.warmup_rounds = warmup_rounds  # 前10轮不用PPO，只用规则/混合
        self.mix_epsilon_start = 1.0  # 早期完全规则
        self.mix_epsilon_end = 0.0  # 退火到纯PPO
        self.mix_epsilon_decay_rounds = 0  # ε 线性退火轮数

        # --- 动作稳定 / 冷却 / TTL ---
        self.delta_eps = getattr(self, "delta_eps", 0.10)  # 剪枝率变更滞回阈值（默认10%）
        self.cooldown_rounds = getattr(self, "cooldown_rounds", 20)  # 至少间隔8轮才允许更换p
        self.pruning_freeze_rounds = getattr(self, "pruning_freeze_rounds", max(5, warmup_rounds // 2))
        self.max_prune_changes_ratio = getattr(self, "max_prune_changes_ratio", 0.2)  # 每轮最多20%客户端变更p
        self.max_prune_changes_per_round = getattr(self, "max_prune_changes_per_round", None)  # 若设定则覆盖ratio
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
        # R2 shaping weights: balance accuracy progress vs. latency fairness.
        self.w_r2_latency = getattr(self, "w_r2_latency", 0.8)
        self.w_r2_acc = getattr(self, "w_r2_acc", 0.6)
        self.w_r2_loss = getattr(self, "w_r2_loss", 0.3)
        self.w_r2_switch = getattr(self, "w_r2_switch", 0.2)

        self.rewards = []
        self.client_wait_times = []
        self.round_time_diff = []
        self.total_run_time = 0.
        self.R1_list = []
        self.R2_list = []
        self.loss_list = []
        self._prev_exec_p = [float(getattr(c, "cur_pruning_rate", 0.0)) for c in self.clients]
        self._pending_s1 = None
        self._pending_s2 = None
        self.ema_latency_score = None
        self.ema_latency_alpha = getattr(self, "ema_latency_alpha", 0.1)
        self.prev_train_acc = None
        self.prev_eval_acc = None
        self.prev_avg_loss = None
        self.signal_alpha = getattr(self, "signal_alpha", 0.2)
        self.loss_ema_alpha = getattr(self, "loss_ema_alpha", 0.2)
        self.train_acc_ema = None
        self.eval_acc_ema = None
        self.loss_ema = None
        self.prev_train_signal = None
        self.prev_eval_signal = None

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

        # R2 分项归一化，提升 reward 尺度稳定性并减少项间“抢权重”。
        self.r2_use_term_norm = getattr(self, "r2_use_term_norm", True)
        self.r2_norm_alpha = getattr(self, "r2_norm_alpha", 0.08)
        self.r2_term_clip = getattr(self, "r2_term_clip", 3.0)
        self.r2_time_norm = _EmaNorm(self.r2_norm_alpha)
        self.r2_acc_norm = _EmaNorm(self.r2_norm_alpha)
        self.r2_loss_norm = _EmaNorm(self.r2_norm_alpha)
        self.switch_pen_ema = 0.0
        self.switch_pen_alpha = getattr(self, "switch_pen_alpha", 0.1)

        # 进展/水平项需要的基线
        self.ema_acc = 0.0
        self.best_acc = 0.0
        # ======================================================

    @staticmethod
    def _ema_update(old_value, new_value, alpha):
        x = float(new_value)
        if old_value is None:
            return x
        a = float(np.clip(alpha, 1e-4, 1.0))
        return float((1.0 - a) * float(old_value) + a * x)

    def _normalize_r2_component(self, key, value):
        v = float(value)
        if not self.r2_use_term_norm:
            return v
        if key == "time":
            normer = self.r2_time_norm
        elif key == "acc":
            normer = self.r2_acc_norm
        elif key == "loss":
            normer = self.r2_loss_norm
        else:
            return v
        normer.update(v)
        z = normer.norm(v)
        return float(np.clip(z, -self.r2_term_clip, self.r2_term_clip))

    def _normalize_switch_penalty(self, switch_penalty):
        p = float(max(0.0, switch_penalty))
        self.switch_pen_ema = self._ema_update(self.switch_pen_ema, p, self.switch_pen_alpha)
        scale = max(float(self.switch_pen_ema), 1e-6)
        return float(np.clip(p / scale, 0.0, self.r2_term_clip))

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
        return True

    def _mark_changed(self, client_id):
        self.client_last_change[client_id] = self.round_id

    def _normalize_candidate_p(self, x, bins=None):
        x = float(x)
        if bins is not None:
            idx = int(torch.argmin(torch.abs(bins - x)).item())
            x = float(bins[idx].item())
        else:
            x = float(np.clip(x, float(self.agent.cfg.p_low), float(self.agent.cfg.p_high)))
        return x

    def _build_stage1_action_mask(self, p_exec, bins):
        """
        Build per-client feasible prune-bin mask before sampling policy actions.
        This makes sampled action closer to actual executable action space.
        """
        if bins is None:
            return None
        bins = bins.to(self.device).float()
        N = len(self.clients)
        B = int(bins.numel())
        mask = torch.zeros((N, B), dtype=torch.bool, device=self.device)
        freeze = self.round_id < self.pruning_freeze_rounds

        for i, client in enumerate(self.clients):
            old_p = float(p_exec[i])
            keep_j = int(torch.argmin(torch.abs(bins - old_p)).item())
            # Keep-current action is always feasible.
            mask[i, keep_j] = True
            if freeze:
                continue
            for j in range(B):
                if j == keep_j:
                    continue
                cand = float(bins[j].item())
                if self._allow_change(client.id, old_p, cand):
                    mask[i, j] = True
            if not bool(mask[i].any()):
                mask[i, keep_j] = True
        return mask

    def _stabilize_pruning_actions(self, p_proposed, bins=None):
        """
        对 proposed p 施加稳定约束：
        1) 冻结期不改；
        2) 冷却 + 滞回；
        3) 每轮最多 K 个客户端改 p（按 |Δp| 从大到小选）。
        """
        N = len(self.clients)
        old_ps = [float(getattr(c, "cur_pruning_rate", p_proposed[i])) for i, c in enumerate(self.clients)]
        if self.round_id < self.pruning_freeze_rounds:
            return old_ps, []

        candidates = []
        for i, client in enumerate(self.clients):
            cand = self._normalize_candidate_p(p_proposed[i], bins=bins)
            if self._allow_change(client.id, old_ps[i], cand):
                candidates.append((abs(cand - old_ps[i]), i, cand))

        if self.max_prune_changes_per_round is not None:
            k = int(max(0, self.max_prune_changes_per_round))
        else:
            k = int(np.ceil(float(self.max_prune_changes_ratio) * max(N, 1)))
        k = max(1, k) if N > 0 else 0
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[:k]

        final_ps = list(old_ps)
        changed = []
        for _, idx, cand in chosen:
            final_ps[idx] = cand
            self._mark_changed(self.clients[idx].id)
            changed.append(idx)
        return final_ps, changed

    @staticmethod
    def _minmax_norm(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        v_min = float(arr.min())
        v_max = float(arr.max())
        if v_max - v_min < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - v_min) / (v_max - v_min)

    @staticmethod
    def _mean_norm(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        return arr / max(float(arr.mean()), 1e-6)

    def _build_s1_state(self, t_norm, entropies, sizes, p_exec):
        h_norm = self._minmax_norm(entropies)
        d_norm = self._mean_norm(sizes)
        p_vec = np.asarray(p_exec, dtype=np.float32)
        feats = np.stack([t_norm, h_norm, d_norm, p_vec], axis=1)
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def _build_s2_state(self, t_norm, entropies, sizes, p_next):
        h_norm = self._minmax_norm(entropies)
        d_norm = self._mean_norm(sizes)
        p_vec = np.asarray(p_next, dtype=np.float32)
        tm_vec = (1.0 + (1.0 - p_vec)) * t_norm
        feats = np.stack([tm_vec, h_norm, d_norm, p_vec], axis=1)
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

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
        # ===== A) 收集指标（这一步对应“执行了上一轮下发动作”） =====
        times, entropies, sizes, do_times = [], [], [], []
        train_accs = []
        p_exec, E_exec, losses = [], [], []

        for client in self.clients:
            train_acc, total_time, entropy, local_data_size, _, \
                client_last_pruning_rate, client_epochs, avg_loss, client_do_time = client.feddra_do()

            train_accs.append(float(train_acc))
            times.append(float(total_time))
            entropies.append(float(entropy))
            sizes.append(int(local_data_size))
            p_exec.append(float(client_last_pruning_rate))
            E_exec.append(int(client_epochs))
            losses.append(float(avg_loss))
            do_times.append(float(client_do_time))

        N = len(self.clients)
        if N == 0:
            return {}
        train_acc_mean = float(np.mean(train_accs)) if train_accs else 0.0
        eval_accs = [float(getattr(c, "last_acc", 0.0)) for c in self.clients]
        eval_acc_lag = float(np.mean(eval_accs)) if eval_accs else 0.0

        total_client_wait_time = self.cal_wait_time(do_times)
        self.client_wait_times.append(total_client_wait_time)

        # ===== B) 构造当前状态 =====
        T_epoch = np.array([t / max(e, 1) for t, e in zip(times, E_exec)], dtype=np.float32)
        Tmin = float(T_epoch.min()) if len(T_epoch) else 1.0
        t_norm = (T_epoch / max(Tmin, 1e-6)).astype(np.float32)
        s1_cur = self._build_s1_state(t_norm, entropies, sizes, p_exec)

        # ===== C) 奖励（对应上一轮动作） =====
        ratio_mm = (T_epoch.max() / max(T_epoch.min(), 1e-6)) if len(T_epoch) else 1.0
        var_T = np.var(T_epoch / max(T_epoch.mean(), 1e-6)) if len(T_epoch) else 0.0
        prev_exec = np.asarray(self._prev_exec_p if len(self._prev_exec_p) == N else p_exec, dtype=np.float32)
        delta_p = float(np.abs(np.asarray(p_exec, dtype=np.float32) - prev_exec).mean()) if len(p_exec) else 0.0
        self._prev_exec_p = [float(x) for x in p_exec]

        R1_base = np.exp(-(ratio_mm - 1.0))
        R1 = float(np.clip(R1_base - 0.3 * var_T - 0.1 * delta_p, -1.0, 1.0))

        eff_times = np.asarray(do_times, dtype=np.float32)
        train_times = np.asarray(times, dtype=np.float32)
        time_span = float(eff_times.max() - eff_times.min()) if eff_times.size > 0 else 0.0
        mean_time = float(eff_times.mean()) if eff_times.size > 0 else 1.0
        span_ratio = time_span / max(mean_time, 1e-6)

        # 剪枝导致的额外开销（只统计正值），用于把“频繁改p”带来的耗时反映到 R2
        prune_overhead = np.clip(eff_times - train_times, a_min=0.0, a_max=None)
        prune_ratio = float(prune_overhead.mean() / max(mean_time, 1e-6)) if eff_times.size > 0 else 0.0

        # latency_index 越大越差；由 time_diff 与 prune_overhead 共同构成
        latency_index = float(span_ratio + 0.5 * prune_ratio)

        # EMA 基线：奖励关注“是否比近期更快”，避免长期饱和在常数附近
        if self.ema_latency_score is None:
            self.ema_latency_score = latency_index
        else:
            a = float(np.clip(self.ema_latency_alpha, 1e-4, 1.0))
            self.ema_latency_score = (1.0 - a) * self.ema_latency_score + a * latency_index

        improve = (self.ema_latency_score - latency_index) / max(abs(self.ema_latency_score), 1e-6)
        avg_loss = sum(losses) / max(len(losses), 1)
        time_term = float(np.tanh(improve) - np.tanh(0.5 * latency_index))

        # 用 EMA 融合 train/eval 信号，减弱 eval_lag 对 reward 的错配影响。
        self.train_acc_ema = self._ema_update(self.train_acc_ema, train_acc_mean, self.signal_alpha)
        if eval_acc_lag > 0.0 or self.eval_acc_ema is not None:
            self.eval_acc_ema = self._ema_update(self.eval_acc_ema, eval_acc_lag, self.signal_alpha)

        train_signal = float(0.6 * train_acc_mean + 0.4 * self.train_acc_ema)
        eval_signal = float(0.4 * eval_acc_lag + 0.6 * (self.eval_acc_ema if self.eval_acc_ema is not None else eval_acc_lag))

        if self.prev_train_signal is None:
            delta_train_acc = 0.0
        else:
            delta_train_acc = float(train_signal - self.prev_train_signal)

        if self.prev_eval_signal is None:
            delta_eval_acc = 0.0
        else:
            delta_eval_acc = float(eval_signal - self.prev_eval_signal)

        # progress + generalization gap 抑制，避免 train_acc 虚高误导 R2。
        gap_pen = float(np.tanh(max(0.0, train_signal - eval_signal) / 8.0))
        acc_term = float(
            0.55 * np.tanh(delta_train_acc / 1.5)
            + 0.45 * np.tanh(delta_eval_acc / 1.0)
            - 0.15 * gap_pen
        )

        prev_loss_ema = self.loss_ema
        self.loss_ema = self._ema_update(self.loss_ema, avg_loss, self.loss_ema_alpha)
        if prev_loss_ema is None:
            loss_term = 0.0
        else:
            rel_loss_improve = (prev_loss_ema - avg_loss) / max(abs(prev_loss_ema), 1e-6)
            loss_term = float(np.tanh(rel_loss_improve))

        switch_penalty = float(np.clip(delta_p / max(self.delta_eps, 1e-6), 0.0, 2.0))
        time_comp = self._normalize_r2_component("time", time_term)
        acc_comp = self._normalize_r2_component("acc", acc_term)
        loss_comp = self._normalize_r2_component("loss", loss_term)
        switch_comp = self._normalize_switch_penalty(switch_penalty)
        R2_raw = (
            self.w_r2_latency * time_comp
            + self.w_r2_acc * acc_comp
            + self.w_r2_loss * loss_comp
            - self.w_r2_switch * switch_comp
        )
        R2 = float(np.clip(R2_raw, -1.5, 1.0))

        self.prev_train_signal = train_signal
        if eval_signal > 0.0:
            self.prev_eval_signal = eval_signal
        self.prev_train_acc = train_acc_mean
        if eval_acc_lag > 0.0:
            self.prev_eval_acc = eval_acc_lag
        self.prev_avg_loss = avg_loss

        self.R1_list.append(R1)
        self.R2_list.append(R2)
        self.loss_list.append(avg_loss)
        self.round_time_diff.append(time_span)
        print(f"round {self.round_id}  R1 {R1:.4f}  R2 {R2:.4f} loss {avg_loss:.6f}")
        print(
            f"[Round {self.round_id}] timing: time_diff={time_span:.4f}, "
            f"span_ratio={span_ratio:.4f}, prune_ratio={prune_ratio:.4f}, latency_idx={latency_index:.4f}"
        )
        print(
            f"[Round {self.round_id}] reward-shaping: train_acc={train_acc_mean:.2f}, eval_acc_lag={eval_acc_lag:.2f}, "
            f"eval_signal={eval_signal:.2f}, "
            f"d_train_acc={delta_train_acc:.3f}, d_eval_acc={delta_eval_acc:.3f}, "
            f"acc_term={acc_term:.4f}, loss_term={loss_term:.4f}, switch_pen={switch_penalty:.4f}, "
            f"time_comp={time_comp:.4f}, acc_comp={acc_comp:.4f}, loss_comp={loss_comp:.4f}, switch_comp={switch_comp:.4f}, "
            f"R2_raw={R2_raw:.4f}"
        )

        # ===== D) 规则策略（warmup 用） =====
        p_rule, E_rule = self._rule_policy(times, entropies)
        use_ppo = self.round_id >= self.warmup_rounds

        use_bins = getattr(self.agent.cfg.s1, "use_discrete_bins", True)
        bins = self.agent.prune_bins.detach().cpu() if use_bins else None
        E_min = int(getattr(self.agent.cfg, "E_min", 1))
        E_max = int(getattr(self.agent.cfg, "E_max", 19))

        cur_s1_pack = None
        cur_s2_pack = None
        changed_idx = []
        overridden_cnt = 0

        # ===== E) 从当前状态产生下一轮动作 =====
        if use_ppo:
            s1_action_mask = self._build_stage1_action_mask(p_exec, self.agent.prune_bins if use_bins else None)
            with torch.no_grad():
                out1 = self.agent.select_pruning(s1_cur, valid_mask=s1_action_mask)
            p_sampled = out1["p"].squeeze(-1).detach().cpu().numpy().tolist()

            p_next, changed_idx = self._stabilize_pruning_actions(p_sampled, bins=bins if use_bins else None)
            overridden_cnt = int(np.sum(np.abs(np.asarray(p_next, dtype=np.float32) - np.asarray(p_sampled, dtype=np.float32)) > 1e-6))

            s2_cur = self._build_s2_state(t_norm, entropies, sizes, p_next)
            with torch.no_grad():
                out2 = self.agent.select_epochs(s2_cur, k=N, E_min=E_min, E_max=E_max)
            E_next = [int(x) for x in out2["E"].tolist()]

            cur_s1_pack = dict(
                s=s1_cur.detach(),
                a=out1["a"].detach(),
                logp=out1["logp"].detach(),
                v=out1["v"].detach(),
                mask=out1.get("mask").detach() if out1.get("mask") is not None else None,
            )
            cur_s2_pack = dict(
                s=s2_cur.detach(),
                a=out2["a"].detach(),
                logp=out2["logp"].detach(),
                v=out2["v"].detach(),
                budget=int(out2["residual_budget"]),
                seq=out2.get("seq", None).detach() if out2.get("seq", None) is not None else None,
                cap=int(out2.get("cap_per_client", max(0, E_max - E_min))),
            )
        else:
            p_next, changed_idx = self._stabilize_pruning_actions(p_rule, bins=bins if use_bins else None)
            E_next = [int(np.clip(int(e), E_min, E_max)) for e in E_rule]
            s2_cur = self._build_s2_state(t_norm, entropies, sizes, p_next)

        # ===== F) 把“上一轮动作”与“本轮回报/下一状态”写入 buffer =====
        if self._pending_s1 is not None:
            self.agent.store_transition_stage1(
                s=self._pending_s1["s"],
                a=self._pending_s1["a"],
                logp=self._pending_s1["logp"],
                v=self._pending_s1["v"],
                r=R1,
                s_next=s1_cur.detach(),
                done=0,
                mask=self._pending_s1.get("mask", None),
            )
        if self._pending_s2 is not None:
            self.agent.store_transition_stage2(
                s=self._pending_s2["s"],
                a=self._pending_s2["a"],
                logp=self._pending_s2["logp"],
                v=self._pending_s2["v"],
                r=R2,
                s_next=s2_cur.detach(),
                done=0,
                residual_budget=int(self._pending_s2["budget"]),
                seq=self._pending_s2.get("seq", None),
                cap_per_client=int(self._pending_s2.get("cap", max(0, E_max - E_min))),
            )

        # ===== G) 下发下一轮动作 =====
        print(
            f"[Round {self.round_id}] pruning-change summary: changed={len(changed_idx)}/{N}, "
            f"freeze={self.pruning_freeze_rounds}, cooldown={self.cooldown_rounds}, "
            f"delta_eps={self.delta_eps:.3f}, overridden={overridden_cnt}"
        )
        for i, client in enumerate(self.clients):
            Ei = int(np.clip(int(E_next[i]), E_min, E_max))
            client.cur_pruning_rate = float(p_next[i])
            client.training_intensity = Ei
            print(
                f"[Round {self.round_id}] Client {client.id} -> next p={client.cur_pruning_rate:.4f}, E={client.training_intensity}"
            )

        # 仅在使用 PPO 动作时，保留当前动作作为“下一轮待结算 transition”
        self._pending_s1 = cur_s1_pack
        self._pending_s2 = cur_s2_pack

        # ===== H) 条件更新 PPO =====
        self.round_id += 1
        if (
            self.round_id > self.warmup_rounds
            and len(self.agent.buf1) >= self.ppo_update_every
            and len(self.agent.buf2) >= self.ppo_update_every
        ):
            out_s1 = self.agent.ppo_update_stage1()
            out_s2 = self.agent.ppo_update_stage2()
            print("PPO1 updated ✅", out_s1)
            print("PPO2 updated ✅", out_s2)

        return {
            c.id: dict(
                pruning_rate=float(getattr(c, "cur_pruning_rate", 0.0)),
                epochs=int(getattr(c, "training_intensity", 1)),
            )
            for c in self.clients
        }

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
        self.round_time_diff.append(max(client_do_time) - min(client_do_time))


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
        self.round_time_diff.append(max(client_do_time) - min(client_do_time))


    def fedprox_aggregate(self, client_models, weights=None):
        """
        FedProx 的聚合与 FedAvg 相同（通常使用样本数加权）。
        client_models: [client.model.state_dict(), ...]
        """
        server_model = self.server_model.to(self.device)
        state = {k: torch.zeros_like(v, device=self.device) for k, v in server_model.state_dict().items()}

        if weights is None:
            weights = [1.0 / max(len(client_models), 1) for _ in client_models]

        for w, sd in zip(weights, client_models):
            for k, v in sd.items():
                if k in state:
                    state[k] += w * v.to(self.device)

        server_model.load_state_dict(state, strict=False)
        self.server_model = server_model

        # 同步回客户端
        sync_state = {k: v.detach().cpu() for k, v in self.server_model.state_dict().items()}
        for c in self.clients:
            c.model.load_state_dict(sync_state, strict=False)


    def fedprox_do(self, local_epochs=None, mu=None, lr=0.1, momentum=0.9, weight_decay=0.0):
        """
        单轮 FedProx：所有客户端并行(顺序)本地训练 -> 聚合 -> 回写
        """
        start = time.time()
        if local_epochs is None:
            local_epochs = int(getattr(self, "fedprox_local_epochs", 5))
        if mu is None:
            mu = float(getattr(self, "fedprox_mu", 0.01))

        # 下发当前 server 参数
        server_state = {k: v.detach().cpu() for k, v in self.server_model.state_dict().items()}

        client_states = []
        times = []
        do_times = []


        # 本地训练
        for client in self.clients:
            # 如果你想全局统一 E，这里可以覆盖：client.training_intensity = local_epochs
            ret = client.fedprox_do(server_state, epochs=local_epochs, mu=mu,
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
            # ret 同 feddra_do 的 9 元组
            _, total_time, _, _, _, _, _, avg_loss, do_time = ret
            times.append(float(total_time))
            do_times.append(float(do_time))

            client_states.append({k: v.detach().cpu() for k, v in client.model.state_dict().items()})

        # 样本数加权
        self.fedprox_aggregate(client_states, weights=None)

        # 统计等待时间（和你现有的 cal_wait_time 保持一致）
        total_client_wait_time = self.cal_wait_time(do_times)
        self.client_wait_times.append(total_client_wait_time)

        self.total_run_time += (time.time() - start)

    def cal_wait_time(self, total_time):
        # 根据client使用的total_time计算客户端等待的时间
        max_time = max(total_time)
        wait_time = sum(max_time - t for t in total_time)
        return wait_time
