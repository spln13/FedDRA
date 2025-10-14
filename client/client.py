import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mini_vgg import MiniVGG
from dataset.fed_dirichlet_io import NpzArrayDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Client(object):
    def __init__(self, client_id, device, model_name, training_intensity, dataset_name, batch_size=16, s=0.001, base_dir="./dataset"):
        self.id = client_id
        self.device = device
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_intensity = training_intensity
        self.batch_size = batch_size
        self.s = s
        self.last_pruning_rate = 1.
        self.cur_pruning_rate = 1.
        self.training_epochs_for_prune = 10
        self.model = self._build_model()  # client自己的模型
        self.aggregated_model = self._build_model()  # 聚合后的全局模型
        self.last_acc = 0.
        self.base_dir = base_dir
        self.round = 0

    def do(self):
        # 每一轮联邦学习循环，client的do，由主程序调用，要做的事情
        # 0. 判断剪枝率是否一致，一致则使用mask从aggregated_model汇总获取剪枝后的模型，不一致则从aggregated_model中获取
        # 1. 获取从server获取的剪枝后模型，和训练轮数
        # 2. 开始本地训练epochs轮次，统计for ppo的指标
        # 3. 需要返回的指标:
        if self.cur_pruning_rate != self.last_pruning_rate:
            # 剪枝率不一致，需要重新剪枝
            print("[client{}, round{}] pruning rate changed from {:.2f} to {:.2f}, need to prune the model.".format(self.id, self.round, self.last_pruning_rate, self.cur_pruning_rate))
            self.train_for_prune()  # 使用本地数据训练一下全局模型，更新bn参数
            self.model = self.prune(self.aggregated_model, self.cur_pruning_rate)


        acc, total_time, avg_loss, entropy, local_data_size = self.train()
        self.last_pruning_rate = self.cur_pruning_rate
        print("[client{}, round{}] finished training, acc: {:.2f}, time: {:.2f}, avg_loss: {:.6f}, entropy: {:.6f}, "
              "local_data_size: {}, pruning_rate: {:.2f}, training_intensity: {}".format(self.round, self.id, acc,
                                                                                         total_time, avg_loss,
                                                                                         entropy, local_data_size,
                                                                                         self.cur_pruning_rate,
                                                                                         self.training_intensity))
        self.round += 1
        return acc, total_time, entropy, local_data_size, self.id, self.cur_pruning_rate, self.training_intensity, avg_loss

    def _build_model(self):
        if self.model_name == 'MiniVGG':
            model = MiniVGG(dataset=self.dataset_name)
            return model
        else:
            raise NotImplementedError

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


    def prune(self, model, pruning_rate):
        """从一个完整模型剪枝到剪枝率=pruning_rate模型"""
        mask = model.mask
        total = 0
        for layer_mask in mask:
            total += len(layer_mask)
        bn = torch.zeros(total)  # bn用于存储模型中所有bn层中缩放因子的绝对值
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:index + size] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)  # 对缩放因子 升序排列
        threshold_index = int(total * pruning_rate)
        threshold = y[threshold_index]  # 获得缩放因子门槛值，低于此门槛值的channel被prune掉
        print("threshold: {:.4f}".format(threshold))
        pruned = 0
        new_cfg = []  # 每个bn层剩余的通道数或者是maxpooling层, 用于初始化模型
        new_cfg_mask = []  # 存储layer_mask数组
        layer_index = 0  # 当前layer下标 当前层为batchnorm才++
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                layer_mask = weight_copy.ge(threshold).float()  # 01数组
                indices = [i for i, x in enumerate(mask[layer_index]) if x == 1.0]  # 获取之前mask中所有保留通道的下标
                if torch.sum(layer_mask) == 0:  # 如果所有通道都被剪枝了，则保留权重最大的一个通道
                    _, idx = torch.max(weight_copy, 0)
                    layer_mask[idx.item()] = 1.0
                pruned += layer_mask.shape[0] - torch.sum(layer_mask)
                m.weight.data.mul_(layer_mask)
                m.bias.data.mul(layer_mask)
                idx = 0
                for _, tag in enumerate(layer_mask):
                    if tag == 0.:  # 该通道应该被剪枝
                        old_mask_index = indices[idx]  # 获取对应之前mask中的下标
                        idx += 1
                        mask[layer_index][old_mask_index] = 0.  # 将mask中对应通道
                new_cfg.append(int(torch.sum(layer_mask)))
                new_cfg_mask.append(layer_mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, layer_mask.shape[0], int(torch.sum(layer_mask))))
                layer_index += 1
            elif isinstance(m, nn.MaxPool2d):
                new_cfg.append('M')
        model.cfg = new_cfg
        model.mask = new_cfg_mask
        new_model = MiniVGG(cfg=new_cfg, dataset=self.dataset_name).to(self.device)
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)  # 当前layer_id的层开始时的通道 cifar初始三个输入通道全部保留
        end_mask = new_cfg_mask[layer_id_in_cfg]  # 当前layer_id的层结束时的通道
        for [m0, m1] in zip(model.modules(), new_model.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(
                    np.argwhere(np.asarray(end_mask.cpu().numpy())))  # idx1是end_mask值非0的下标 squeeze()转换为1维数组
                if idx1.size == 1:  # 若只有一个元素则会成为标量，需要转成数组
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(new_cfg_mask):  # do not change in Final FC
                    end_mask = new_cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # [out_channels, int_channels, H, W]
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()

        return new_model



    def train_for_prune(self, sr=True):
        # 训练少轮次，更新bn参数供剪枝算法使用
        model = self.aggregated_model
        model.fill_bn()
        epochs = self.training_epochs_for_prune  # for剪枝训练强度
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []  # ✅ 用于存储所有batch的loss
        s = self.s

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
                train_loader_tqdm.set_description(f'Train For Prune Epoch: {epoch} Loss: {loss.item():.6f}')

        self.aggregated_model = model

    def test(self):
        model = self.load_model()
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


    def _npz_path(self, split: str) -> str:
        return os.path.join(self.base_dir, self.dataset_name, split, f"{self.id}.npz")

    def load_train_data(self, batch_size: int = 64, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("train"), self.dataset_name, train=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def load_test_data(self, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("test"), self.dataset_name, train=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    def load_local_test_data(self, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
        ds = NpzArrayDataset(self._npz_path("local_test"), self.dataset_name, train=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
