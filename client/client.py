import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mini_vgg import MiniVGG
from utils.utils import read_client_data
from torch.utils.data import DataLoader
from tqdm import tqdm


class Client(object):
    def __init__(self, client_id, device, model_name, training_intensity, dataset, pr_list, batch_size=16, s=0.0001):
        self.id = client_id
        self.device = device
        self.model_name = model_name
        self.dataset = dataset
        self.training_intensity = training_intensity
        self.batch_size = batch_size
        self.s = s
        self.current_pr = 1.
        self.last_pruning_rate = 0.
        self.cur_pruning_rate = 0.
        self.model = self._build_model()  # client自己的模型
        self.aggregated_model = self._build_model()  # 聚合后的全局模型
        self.last_acc = 0.

    def do(self):
        # 每一轮联邦学习循环，client的do，由主程序调用，要做的事情
        # 1. 获取从server获取的剪枝后模型，和训练轮数
        # 2. 开始本地训练epochs轮次，统计for ppo的指标
        # 3. 需要返回的指标:
        acc, total_time, avg_loss, entropy, local_data_size = self.train()
        self.last_pruning_rate = self.cur_pruning_rate
        return acc, total_time, entropy, local_data_size, self.id, self.cur_pruning_rate, self.training_intensity, avg_loss

    def _build_model(self):
        if self.model_name == 'MiniVGG':
            model = MiniVGG(dataset=self.dataset)
            return model
        else:
            raise NotImplementedError

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, is_local_test=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=False, is_local_test=is_local_test)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_model(self):
        return self.model

    def train(self, sr=False):
        """模型训练制定epoch，需要统计训练时间、平均loss、本地数据量Di和信息熵"""
        # ---------- 工具函数：预测熵 ----------
        def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
            prob = F.softmax(logits, dim=-1).clamp_min(1e-12)
            return -(prob * prob.log()).sum(dim=-1)

        @torch.no_grad()
        def _predictive_entropy_loader(model: torch.nn.Module,
                                       loader,
                                       device: str = "cpu",
                                       max_batches: int = 2) -> float:
            """计算预测熵（默认仅前 max_batches 个batch）"""
            model.eval()
            total_H, total_n = 0.0, 0
            for b_idx, (xb, _) in enumerate(loader):
                xb = xb.to(device)
                logits = model(xb)
                H_b = _entropy_from_logits(logits)  # [B]
                total_H += float(H_b.sum().item())
                total_n += xb.shape[0]
                if (max_batches is not None) and (b_idx + 1 >= max_batches):
                    break
            return total_H / max(total_n, 1)

        # ---------- 原有训练逻辑 ----------
        model = self.load_model()
        epochs = self.training_intensity  # 训练强度
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []  # ✅ 用于存储所有batch的loss
        s = self.s
        start_time = time.time()  # 记录训练开始时间

        # ✅ 统计本地数据量 Di（样本总数）
        local_dataset = getattr(train_loader, "dataset", None)
        local_data_size = len(local_dataset) if local_dataset is not None else 0

        for epoch in range(epochs):
            if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            # training
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())  # ✅ 记录每个batch的loss

                loss.backward()
                if sr:  # update batchnorm
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1
                optimizer.step()
                train_loader_tqdm.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')

        end_time = time.time()
        total_time = end_time - start_time

        # ✅ 计算平均loss
        avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        print(f"[client{self.id}] Average training loss: {avg_loss:.6f}")

        # ✅ 计算信息熵（取前2个batch做近似）
        entropy = _predictive_entropy_loader(model, train_loader, device=self.device, max_batches=2)
        print(f"[client{self.id}] Predictive entropy: {entropy:.6f}")

        # ✅ 本地测试集acc
        acc = self.local_test(model)
        print(f"[client{self.id} accuracy] accuracy: {acc:.4f}")

        self.model = model
        # ✅ 返回 acc、训练时间、平均loss、信息熵、本地数据量Di
        return acc, total_time, avg_loss, entropy, local_data_size

    def first_evaluate(self):
        """
        算法开始时，训练模型得到初始模型准确度和训练时间，for后续PPO得到初始模型剪枝率和训练轮数
        """
        model = self.load_model()
        # train model 50 轮
        epoch = 1
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []
        start_time = time.time()
        for epoch in range(epoch):
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                train_loader_tqdm.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')
        end_time = time.time()
        training_time = end_time - start_time

        # 在client本地测试集跑一下得到acc一并返回给server
        acc = self.local_test(model)

        return acc, training_time

    def local_test(self, model) -> float:
        # client使用本地小测试集进行测试，返回准确率供PPO模型参考
        test_loader = self.load_test_data(batch_size=128)
        model = model.to(self.device)
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                data, target = data, target
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('\nclient_id: {}, Test acc: {}/{} ({:.1f}%)\n'.format(self.id, correct,
                                                                    len(test_loader.dataset),
                                                                    100. * correct / len(
                                                                        test_loader.dataset)))
        acc = 100. * correct / len(test_loader.dataset)
        return acc



