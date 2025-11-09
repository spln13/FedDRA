import argparse
import time

import torch

from server.server import Server
from client.client import Client


def fedAvg(args):
    # 这里弄fedavg的算法流程
    client_nums = 10
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fl_rounds = 200
    clients = []
    epochs = args.epochs
    max_acc = 0.
    for i in range(client_nums):
        client = Client(i, device, model_name, epochs, dataset, 16, batch_norm=False)
        clients.append(client)
    server = Server(device, clients, dataset, model_name, batch_norm=False)
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
    final_acc = []
    for c in clients:
        acc = c.test()
        final_acc.append(acc)
        print("Client {} Test Acc: {:.2f}%".format(c.id, acc))

    print("#######Max Acc: {:.2f}%".format(max(acc_list)))
    print("#######Final Average Acc: {:.2f}%".format(sum(final_acc) / len(final_acc)))
    cal_run_time(server)
    print_time_diffs(server.round_time_diff)
    print_loss(server.loss_list)
    print_acc_list(acc_list)


def fedBN(args):
    # 这里弄fedavg的算法流程
    client_nums = 10
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fl_rounds = 200
    clients = []
    epochs = args.epochs
    max_acc = 0.
    for i in range(client_nums):
        client = Client(i, device, model_name, epochs, dataset, 16, batch_norm=True)
        clients.append(client)
    server = Server(device, clients, dataset, model_name, batch_norm=True)
    print("fedAvg Start Training...")
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        accs = []
        server.fedbn_do()
        for c in clients:
            acc = c.test()
            accs.append(acc)
        print("Round {} Test Acc: {:.2f}%".format(r, sum(accs) / len(accs)))
        max_acc = max(max_acc, sum(accs) / len(accs))
    final_acc = []
    for c in clients:
        acc = c.test()
        final_acc.append(acc)
        print("Client {} Test Acc: {:.2f}%".format(c.id, acc))

    print("#######Max Acc: {:.2f}%".format(max(final_acc)))
    print("#######Final Average Acc: {:.2f}%".format(sum(final_acc) / len(final_acc)))
    cal_run_time(server)



def fedProx(args):
    server = Server(args)     # 你的 Server 初始化里可将 mu/epochs 存起来
    server.fedprox_mu = args.prox_mu
    server.fedprox_local_epochs = args.local_epochs

    # 训练若干轮
    for r in range(args.rounds):
        print(f"--- FedProx Round {r} ---")
        server.fedprox_do(local_epochs=args.local_epochs, mu=args.prox_mu)



def fedDRA(args):
    client_nums = 10
    fl_rounds = 500
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    pruning_ablation = True  # 是否进行剪枝率消融实验
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ppo related
    d_glob = 6  # 全局特征维度
    d_cli = 7  # client侧特征维度
    p_low = 0.2  # 最低剪枝率
    p_high = 0.8  # 最高剪枝率
    E_min = 3  # 最小训练轮次
    E_max = 17  # 最大训练轮次
    hidden = 256  # PPO网络隐藏层维度
    prune_bins = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
    clients = []
    max_acc = 0.
    for i in range(client_nums):
        client = Client(i, device, model_name, 1, dataset, 16)
        clients.append(client)
    server = Server(device, clients, dataset, model_name, prune_bins, E_min, E_max, batch_norm=True, warmup_rounds=0)
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
        choices=["fedDRA", "fedAvg", "fedBN"],
        help="选择要运行的算法: fedDRA 或 fedAvg (默认: fedDRA)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="联邦学习本地训练epochs"
    )

    args = parser.parse_args()

    if args.algo.lower() == "fedavg":
        print("Running FedAvg...")
        fedAvg(args)
    elif args.algo.lower() == "feddra":
        print("Running FedDRA...")
        fedDRA(args)
    elif args.algo.lower() == "fedbn":
        print("Running FedBN...")
        fedBN(args)
    else:
        raise ValueError(f"未知算法: {args.algo}")


if __name__ == '__main__':
    main()


# 测试命令
# nohup python main.py --algo fedDRA > logs/fedDRA_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# nohup python main.py --algo fedAvg --epochs 10 > logs/fedAvg_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
# nohup python main.py --algo fedBN --epochs 10 > logs/fedAvg_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

