import argparse
import math
import time

import torch

from server.server import Server
from client.client import Client


def _normalize_dataset(name: str) -> str:
    return str(name).strip().lower()


def _normalize_model(name: str) -> str:
    key = str(name).strip().lower()
    if key in ("minivgg", "mini_vgg"):
        return "MiniVGG"
    if key in ("mnistnet", "mnist_net"):
        return "MNISTNet"
    raise ValueError(f"Unsupported model: {name}")


def _build_ablation_flags(args) -> dict:
    return {
        "reward_norm": bool(getattr(args, "disable_reward_norm", False)),
        "eval_smoothing": bool(getattr(args, "disable_eval_smoothing", False)),
        "adaptive_kl": bool(getattr(args, "disable_adaptive_kl", False)),
        "action_mask": bool(getattr(args, "disable_action_mask", False)),
        "prune_stability": bool(getattr(args, "disable_prune_stability", False)),
    }


def _build_standard_context(args, batch_norm=False, default_rounds=500, default_epochs=None):
    client_nums = args.clients
    model_name = args.model
    dataset = args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fl_rounds = args.rounds if args.rounds is not None else default_rounds
    epochs = args.epochs if default_epochs is None else default_epochs
    clients = []
    for i in range(client_nums):
        client = Client(i, device, model_name, epochs, dataset, args.batch_size, batch_norm=batch_norm)
        clients.append(client)
    server = Server(device, clients, dataset, model_name, batch_norm=batch_norm, ablations=_build_ablation_flags(args))
    return clients, server, fl_rounds


def _finalize_standard_run(server, clients, acc_list):
    final_acc = []
    for c in clients:
        acc = c.test()
        final_acc.append(acc)
        print("Client {} Test Acc: {:.2f}%".format(c.id, acc))

    print("#######Max Acc: {:.2f}%".format(max(acc_list)))
    print("#######Final Average Acc: {:.2f}%".format(sum(final_acc) / len(final_acc)))
    cal_run_time(server)
    print_time_diffs(server.round_time_diff)
    print_acc_list(acc_list)


def _assign_resource_adaptive_epochs(clients, base_epochs, min_epochs, max_epochs, ema_epoch_costs):
    client_num = len(clients)
    total_budget = client_num * base_epochs
    if (not ema_epoch_costs) or any(client.id not in ema_epoch_costs for client in clients):
        return [base_epochs for _ in clients]

    inv_costs = [1.0 / max(float(ema_epoch_costs[client.id]), 1e-6) for client in clients]
    inv_sum = sum(inv_costs)
    if inv_sum <= 0.0:
        return [base_epochs for _ in clients]

    raw_epochs = [total_budget * score / inv_sum for score in inv_costs]
    epochs = [int(math.floor(v)) for v in raw_epochs]
    epochs = [max(min_epochs, min(max_epochs, e)) for e in epochs]

    current_budget = sum(epochs)
    residual_order = sorted(
        range(client_num),
        key=lambda idx: (raw_epochs[idx] - math.floor(raw_epochs[idx]), inv_costs[idx]),
        reverse=True,
    )
    while current_budget < total_budget:
        changed = False
        for idx in residual_order:
            if epochs[idx] >= max_epochs:
                continue
            epochs[idx] += 1
            current_budget += 1
            changed = True
            if current_budget >= total_budget:
                break
        if not changed:
            break

    trim_order = sorted(
        range(client_num),
        key=lambda idx: (raw_epochs[idx] - math.floor(raw_epochs[idx]), inv_costs[idx]),
    )
    while current_budget > total_budget:
        changed = False
        for idx in trim_order:
            if epochs[idx] <= min_epochs:
                continue
            epochs[idx] -= 1
            current_budget -= 1
            changed = True
            if current_budget <= total_budget:
                break
        if not changed:
            break

    return epochs


def _update_epoch_cost_ema(clients, ema_epoch_costs, alpha, fallback_epochs):
    for client in clients:
        if not client.client_do_times:
            continue
        last_epochs = max(int(getattr(client, "training_intensity", fallback_epochs)), 1)
        epoch_cost = float(client.client_do_times[-1]) / float(last_epochs)
        prev = ema_epoch_costs.get(client.id)
        if prev is None:
            ema_epoch_costs[client.id] = epoch_cost
        else:
            ema_epoch_costs[client.id] = (1.0 - alpha) * float(prev) + alpha * epoch_cost


def fedAvg(args):
    # 这里弄fedavg的算法流程
    clients, server, fl_rounds = _build_standard_context(args, batch_norm=False, default_rounds=500)
    print("fedAvg Start Training...")
    acc_list = []
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        accs = []
        server.fedavg_do()  # 每一轮的逻辑包在server内实现
        for c in clients:
            acc = c.test()
            accs.append(acc)
        print("Round {} Test Acc: {:.2f}%".format(r, sum(accs) / len(accs)))
        acc_list.append(sum(accs) / len(accs))
    _finalize_standard_run(server, clients, acc_list)


def fedBN(args):
    # 这里弄fedavg的算法流程
    clients, server, fl_rounds = _build_standard_context(args, batch_norm=True, default_rounds=200)
    print("fedBN Start Training...")
    acc_list = []
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        accs = []
        server.fedbn_do()
        for c in clients:
            acc = c.test()
            accs.append(acc)
        round_acc = sum(accs) / len(accs)
        print("Round {} Test Acc: {:.2f}%".format(r, round_acc))
        acc_list.append(round_acc)
    _finalize_standard_run(server, clients, acc_list)



def fedProx(args):
    clients, server, fl_rounds = _build_standard_context(args, batch_norm=False, default_rounds=500)
    server.fedprox_mu = args.prox_mu
    server.fedprox_local_epochs = args.epochs

    print("fedProx Start Training...")
    acc_list = []
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        server.fedprox_do(local_epochs=args.epochs, mu=args.prox_mu, lr=args.prox_lr,
                          momentum=0.9, weight_decay=1e-4)
        accs = []
        for c in clients:
            acc = c.test()
            accs.append(acc)
        round_acc = sum(accs) / len(accs)
        print("Round {} Test Acc: {:.2f}%".format(r, round_acc))
        acc_list.append(round_acc)
    _finalize_standard_run(server, clients, acc_list)


def fedAvgRA(args):
    clients, server, fl_rounds = _build_standard_context(args, batch_norm=False, default_rounds=500)
    ema_epoch_costs = {}
    min_epochs = args.ra_min_epochs
    max_epochs = args.ra_max_epochs if args.ra_max_epochs is not None else max(args.epochs * 2, min_epochs)

    print("fedAvgRA Start Training...")
    print(f"[Resource-Adaptive Config] base_epochs={args.epochs}, min_epochs={min_epochs}, "
          f"max_epochs={max_epochs}, ema_alpha={args.ra_ema_alpha:.2f}")
    acc_list = []
    for r in range(fl_rounds):
        epoch_plan = _assign_resource_adaptive_epochs(
            clients, base_epochs=args.epochs, min_epochs=min_epochs,
            max_epochs=max_epochs, ema_epoch_costs=ema_epoch_costs
        )
        print(f"--- FL Round {r} ---")
        print("[Round {}] resource-adaptive epochs: {}".format(
            r, ", ".join(f"c{client.id}={ep}" for client, ep in zip(clients, epoch_plan))
        ))
        for client, ep in zip(clients, epoch_plan):
            client.training_intensity = int(ep)
        server.fedavg_do()
        accs = []
        for c in clients:
            acc = c.test()
            accs.append(acc)
        round_acc = sum(accs) / len(accs)
        print("Round {} Test Acc: {:.2f}%".format(r, round_acc))
        acc_list.append(round_acc)
        _update_epoch_cost_ema(clients, ema_epoch_costs, args.ra_ema_alpha, args.epochs)

    _finalize_standard_run(server, clients, acc_list)



def fedDRA(args):
    client_nums = args.clients
    fl_rounds = args.rounds if args.rounds is not None else 500
    model_name = args.model
    dataset = args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ppo related
    d_glob = 6  # 全局特征维度
    d_cli = 7  # client侧特征维度
    p_low = 0.2  # 最低剪枝率
    p_high = 0.8  # 最高剪枝率
    E_min = 3  # 最小训练轮次
    E_max = 17  # 最大训练轮次
    hidden = 256  # PPO网络隐藏层维度
    prune_bins = (0.0, 0.1, 0.2, 0.3, 0.4)
    clients = []
    max_acc = 0.
    for i in range(client_nums):
        client = Client(i, device, model_name, 1, dataset, args.batch_size)
        clients.append(client)
    ablations = _build_ablation_flags(args)
    server = Server(
        device, clients, dataset, model_name, prune_bins, E_min, E_max,
        batch_norm=True, warmup_rounds=20, ablations=ablations
    )
    enabled = [k for k, v in ablations.items() if v]
    print(f"[Ablation] disabled modules: {enabled if enabled else 'none'}")
    print("fedDRA Start Training...")
    acc_list = []
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        server.feddra_do()  # 每一轮的逻辑包在server内实现
        accs = []
        for c in clients:
            acc = c.test()
            accs.append(acc)
        print("Round {} Test Acc: {:.2f}%".format(r, sum(accs) / len(accs)))
        acc_list.append(sum(accs) / len(accs))


    print("#######Max Acc: {:.2f}%".format(max(acc_list)))
    cal_run_time(server)
    print_reward(server.R1_list)
    print_reward(server.R2_list)
    print_time_diffs(server.round_time_diff)
    print_loss(server.loss_list)
    print_acc_list(acc_list)


def cal_run_time(server):
    total_run_time = server.total_run_time
    for client in server.clients:
        total_run_time += client.client_total_do_time
    print("#######Total Run Time: {:.2f} seconds".format(total_run_time))
    wait_times = server.client_wait_times
    print("#######Client total wait time: {:.2f} seconds".format(sum(wait_times)))


def print_reward(rewards):
    print("Rewards: ", end="")
    for r in rewards:
        print("{:.4f} ".format(r), end="")
    print()


def print_acc_list(accs):
    print("Accs: ", end="")
    for r in accs:
        print("{:.4f} ".format(r), end="")
    print()


def print_time_diffs(times):
    print("time_diff: ", end="")
    for r in times:
        print("{:.4f} ".format(r), end="")
    print()


def print_loss(losses):
    print("Losses: ", end="")
    for r in losses:
        print("{:.4f} ".format(r), end="")
    print()


def main():
    # 需要加一些参数处理
    parser = argparse.ArgumentParser(description="Federated Learning Runner")
    parser.add_argument(
        "--algo",
        type=str,
        default="fedDRA",
        choices=["fedDRA", "fedAvg", "fedBN", "fedProx", "fedAvgRA"],
        help="选择要运行的算法: fedDRA / fedAvg / fedBN / fedProx / fedAvgRA (默认: fedDRA)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="联邦学习本地训练epochs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="数据集名称，例如: cifar10 / mnist"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mini_vgg",
        help="模型名称，例如: mini_vgg / mnist_net"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="联邦学习总轮数，默认使用各算法内置值（fedDRA/fedAvg=500, fedBN=200）"
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="客户端数量"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="客户端本地训练 batch size"
    )
    parser.add_argument(
        "--prox_mu",
        type=float,
        default=0.01,
        help="FedProx 的 proximal 系数 mu"
    )
    parser.add_argument(
        "--prox_lr",
        type=float,
        default=0.005,
        help="FedProx 本地训练学习率"
    )
    parser.add_argument(
        "--ra_min_epochs",
        type=int,
        default=1,
        help="FedAvgRA 每个客户端的最小本地训练轮数"
    )
    parser.add_argument(
        "--ra_max_epochs",
        type=int,
        default=None,
        help="FedAvgRA 每个客户端的最大本地训练轮数，默认取 2 * epochs"
    )
    parser.add_argument(
        "--ra_ema_alpha",
        type=float,
        default=0.3,
        help="FedAvgRA 对每轮单位 epoch 耗时做 EMA 的平滑系数"
    )
    parser.add_argument(
        "--disable_reward_norm",
        action="store_true",
        help="消融：关闭 R2 分项归一化（time/acc/loss/switch）"
    )
    parser.add_argument(
        "--disable_eval_smoothing",
        action="store_true",
        help="消融：关闭 train/eval EMA 信号平滑，直接使用原始值"
    )
    parser.add_argument(
        "--disable_adaptive_kl",
        action="store_true",
        help="消融：关闭 PPO1 的 adaptive-KL 学习率调整"
    )
    parser.add_argument(
        "--disable_action_mask",
        action="store_true",
        help="消融：关闭 Stage1 可行动作 mask（允许策略采样所有剪枝档）"
    )
    parser.add_argument(
        "--disable_prune_stability",
        action="store_true",
        help="消融：关闭剪枝稳定器（冻结/冷却/滞回/每轮变更上限）"
    )

    args = parser.parse_args()
    args.dataset = _normalize_dataset(args.dataset)
    args.model = _normalize_model(args.model)

    if args.algo.lower() == "fedavg":
        print(f"Running FedAvg... dataset={args.dataset}, model={args.model}")
        fedAvg(args)
    elif args.algo.lower() == "feddra":
        print(f"Running FedDRA... dataset={args.dataset}, model={args.model}")
        fedDRA(args)
    elif args.algo.lower() == "fedbn":
        print(f"Running FedBN... dataset={args.dataset}, model={args.model}")
        fedBN(args)
    elif args.algo.lower() == "fedprox":
        print(f"Running FedProx... dataset={args.dataset}, model={args.model}")
        fedProx(args)
    elif args.algo.lower() == "fedavgra":
        print(f"Running FedAvgRA... dataset={args.dataset}, model={args.model}")
        fedAvgRA(args)
    else:
        raise ValueError(f"未知算法: {args.algo}")


if __name__ == '__main__':
    main()


# 测试命令
# nohup python main.py --algo fedDRA --dataset cifar10 --model mini_vgg > logs/fedDRA_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# nohup python main.py --algo fedAvg --dataset mnist --model mnist_net --epochs 10 > logs/fedAvg_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# nohup python main.py --algo fedBN --dataset mnist --model mnist_net --epochs 10 > logs/fedBN_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# nohup python main.py --algo fedDRA --dataset mnist --model mnist_net --disable_reward_norm --epochs 10 > logs/fedDRA_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

# python main.py --algo fedAvg --dataset mnist --model mnist_net --epochs 10 --rounds 500 > logs/fedDRA_mnist_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# python main.py --algo fedBN  --dataset mnist --model mnist_net --epochs 10 --rounds 500 > logs/fedDRA_mnist_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# python main.py --algo fedProx --dataset cifar10 --model mini_vgg --epochs 10 --rounds 500 --prox_mu 0.01 > logs/fedProx_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# python main.py --algo fedAvgRA --dataset cifar10 --model mini_vgg --epochs 10 --rounds 500 --ra_min_epochs 1 --ra_max_epochs 20 > logs/fedAvgRA_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# python main.py --algo fedDRA --dataset mnist --model mnist_net --rounds 500 --batch_size 16 > logs/fedDRA_mnist_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

# --disable_reward_norm
# --disable_eval_smoothing
# --disable_adaptive_kl
# --disable_action_mask
# --disable_prune_stability
