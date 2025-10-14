import argparse

import torch

from server.server import Server
from client.client import Client


def fedAvg():
    # 这里弄fedavg的算法流程
    client_nums = 20
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    server = Server(device, client_nums, model_name, dataset, batch_norm=False)
    fl_rounds = 500
    clients = []
    for i in range(client_nums):
        client = Client(i, device, model_name, 1, dataset, 16, batch_norm=False)
        clients.append(client)
    server.clients = clients
    print("fedAvg Start Training...")
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        server.fedavg_do()  # 每一轮的逻辑包在server内实现

    final_acc = []
    for c in clients:
        acc = c.test()
        final_acc.append(acc)
        print("Client {} Test Acc: {:.2f}%".format(c.id, acc))

    print("#######Final Average Acc: {:.2f}%".format(sum(final_acc) / len(final_acc)))


def fedDRA():

    client_nums = 20
    fl_rounds = 500
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ppo related
    d_glob = 6  # 全局特征维度
    d_cli = 7  # client侧特征维度
    p_low = 0.2  # 最低剪枝率
    p_high = 0.8  # 最高剪枝率
    E_min = 1  # 最小训练轮次
    E_max = 5  # 最大训练轮次
    hidden = 256  # PPO网络隐藏层维度
    server = Server(device, client_nums, model_name, dataset, d_glob, d_cli, p_low, p_high, E_min, E_max, hidden)
    clients = []
    for i in range(client_nums):
        client = Client(i, device, model_name, 1, dataset, 16)
        clients.append(client)
    server.clients = clients

    print("fedDRA Start Training...")
    for r in range(fl_rounds):
        print(f"--- FL Round {r} ---")
        server.do()  # 每一轮的逻辑包在server内实现

    final_acc = []
    for c in clients:
        acc = c.test()
        final_acc.append(acc)
        print("Client {} Test Acc: {:.2f}%".format(c.id, acc))

    print("#######Final Average Acc: {:.2f}%".format(sum(final_acc) / len(final_acc)))


def main():
    # 需要加一些参数处理
    parser = argparse.ArgumentParser(description="Federated Learning Runner")
    parser.add_argument(
        "--algo",
        type=str,
        default="fedDRA",
        choices=["fedDRA", "fedAvg"],
        help="选择要运行的算法: fedDRA 或 fedAvg (默认: fedDRA)"
    )

    args = parser.parse_args()

    if args.algo.lower() == "fedavg":
        print("Running FedAvg...")
        fedAvg()
    elif args.algo.lower() == "feddra":
        print("Running FedDRA...")
        fedDRA()
    else:
        raise ValueError(f"未知算法: {args.algo}")


if __name__ == '__main__':
    main()
