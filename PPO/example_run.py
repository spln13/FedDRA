# -*- coding: utf-8 -*-
"""
示例：如何把 Dual-Stream PPO 接到联邦训练循环。
你需要把每轮可用的状态 S_glob / S_cli / mask 喂给 agent，
再把动作 p/E 下发给客户端，完成一轮 FL 后返回 reward 等写入缓冲。
累计若干轮后，调用 ppo_update() 更新策略。
"""
import torch
from PPO import DualStreamConfig, DualStreamPPO


def build_fake_state(B, N, d_glob, d_cli):
    """
    根据你的系统构造真实状态。
    这里只是演示：随机张量 + 变长 mask
    """
    S_glob = torch.randn(B, d_glob)
    S_cli  = torch.randn(B, N, d_cli)
    # 构造变长客户端掩码：每个 batch 的有效客户端数量不同
    mask = torch.zeros(B, N)
    for b in range(B):
        k = torch.randint(low=1, high=N+1, size=(1,)).item()
        mask[b, :k] = 1.0
    return S_glob, S_cli, mask


def compute_global_reward(delta_acc, t_max, delta_t, comm_cost,
                          alpha=1.0, beta=0.01, gamma=0.01, lamb=0.0):
    """
    联邦一轮后的全局奖励示例（你可以替换为真实指标）
      R = α·ΔAcc - β·T_max - γ·ΔT - λ·CommCost
    """
    R = alpha * delta_acc - beta * t_max - gamma * delta_t - lamb * comm_cost
    return R


def main():
    # ---- 配置与 agent ----
    B = 2            # 并行联邦群的 batch 数；没有并行就设 1
    N = 5            # 每群最多支持的客户端数（不足用 mask=0 补齐）
    d_glob = 6       # 例如 [Acc, dAcc, Tmax, dT, round_id, bw]
    d_cli = 7        # 例如 [Ti, Hi, |Di|, id_emb, p_prev, E_prev, loss_prev]

    cfg = DualStreamConfig(d_glob=d_glob, d_cli=d_cli,
                           p_low=0.2, p_high=0.9, E_min=1, E_max=5,
                           hidden=256)
    agent = DualStreamPPO(cfg)

    # ---- 模拟若干联邦轮后再更新（on-policy）----
    TRAJ_ROUNDS = 5     # 累计 5 轮后做一次 PPO 更新

    for t in range(TRAJ_ROUNDS):
        # 1) 构造当前轮的状态
        S_glob, S_cli, mask = build_fake_state(B, N, d_glob, d_cli)

        # 2) 用策略采样动作（分配剪枝率与训练轮数）
        with torch.no_grad():
            out = agent.select_actions(S_glob, S_cli, mask)
        p, E, logp = out["p"], out["E"], out["logp"]

        # === 在你真实系统中，这里把 p/E 下发给客户端，执行一轮 FL ===
        # === 客户端训练结束后，你会拿到真实的 Acc、T_max、ΔT、通信等 ===
        # 下面以随机数模拟这些反馈：
        delta_acc = torch.rand(B, 1) * 0.05       # 假设精度提升 0~5%
        t_max = torch.rand(B, 1) * 100 + 200      # 最慢客户端时间（ms/秒，看你单位）
        delta_t = torch.rand(B, 1) * 40           # 客户端间时间差
        comm_cost = torch.rand(B, 1) * 10         # 通信代价（可用平均模型尺寸近似）

        reward = compute_global_reward(delta_acc, t_max, delta_t, comm_cost)  # [B,1]
        done = torch.zeros(B, 1)  # 一般联邦训练不中止

        # 3) 存入缓冲（注意：需要 V；我们再算一次 evaluate 以取到 V）
        eval_out = agent.net.evaluate_actions(S_glob, S_cli, p, E, mask)
        agent.store_transition(dict(
            S_glob=S_glob, S_cli=S_cli, mask=mask,
            p=p, E=E, logp=logp, V=eval_out["V"].detach(),
            reward=reward, done=done
        ))

    # 4) 累计到一定轮数后，执行一次 PPO 更新
    agent.ppo_update(epochs=6, batch_size=64, normalize_adv=True)
    print("PPO update done ✅")


if __name__ == "__main__":
    main()
