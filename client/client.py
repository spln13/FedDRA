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
    def __init__(self, client_id, device, model_name, training_intensity, dataset_name, batch_size=16, s=0.001,
                 base_dir="./dataset", batch_norm=True, pruning_ablation = True):
        self.id = client_id
        self.device = device
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.training_intensity = training_intensity
        self.batch_size = batch_size
        self.s = s
        self.last_pruning_rate = 0.
        self.cur_pruning_rate = 0.
        self.training_epochs_for_prune = 8
        self.finetune_epochs = 10
        self.model = self._build_model(batch_norm=batch_norm)  # client自己的模型
        self.last_acc = 0.
        self.base_dir = base_dir
        self.round = 0
        self.client_do_times = []  # client运行时间数组，包括剪枝时间
        self.client_total_do_time = 0.
        self.batch_norm = batch_norm
        self.pruning_ablation = pruning_ablation
        self.server_model = self._build_model(batch_norm=batch_norm)

    def feddra_do(self):
        # 每一轮联邦学习循环，client的do，由主程序调用，要做的事情
        # 0. 判断剪枝率是否一致，一致则使用mask从aggregated_model汇总获取剪枝后的模型，不一致则从aggregated_model中获取
        # 1. 获取从server获取的剪枝后模型，和训练轮数
        # 2. 开始本地训练epochs轮次，统计for ppo的指标
        # 3. 需要返回的指标:
        prune_time = 0.
        kd_acc = 0.
        pruning_rate_changed = False
        if self.cur_pruning_rate != self.last_pruning_rate:
            # 剪枝率不一致，需要重新剪枝
            pruning_rate_changed = True
            if self.pruning_ablation:
                # 对比空模型知识蒸馏，相同轮数，对比准确率
                # 从全局模型剪枝，然后清空参数，进行知识蒸馏
                # 1) 新建“空”学生（可同步 BN 统计，但不复制卷积/全连接权重）
                student = self._new_minivgg_student(sync_bn_from_server=True, init="kaiming")
                # 2) 先把学生剪到目标比例（保证与“恢复机制”同一目标容量）
                student = self.prune(student, self.cur_pruning_rate)
                # 3) 用 server full 作为 teacher 做 KD
                self.distill_train(student, self.server_model,
                                   distill_epoch=getattr(self, "distill_epochs", 10),
                                   T=getattr(self, "kd_T", 2.0),
                                   alpha=getattr(self, "kd_alpha", 0.7))

                # 4) 测试学生模型的准确率
                kd_acc = self.for_kd_test(student)

            prune_start_time = time.time()
            print("[client{}, round{}] pruning rate changed from {:.2f} to {:.2f}, need to prune the model.".format(
                self.id, self.round, self.last_pruning_rate, self.cur_pruning_rate))
            new_model = self.fill_to_full_model_and_train()  # 使用本地数据训练一下全局模型，更新bn参数
            self.model = self.prune(new_model, self.cur_pruning_rate)
            self.finetune()
            prune_end_time = time.time()
            prune_time = prune_end_time - prune_start_time


        # total_times是用来for PPO决策，client_do_times是用来统计wait_time
        acc, total_time, avg_loss, entropy, local_data_size = self.train()
        total_time = self.mock_time_delay(total_time)
        self.client_do_times.append(total_time + prune_time)
        self.client_total_do_time += total_time + prune_time
        if self.pruning_ablation and pruning_rate_changed:
            acc = self.test()
            # 打印acc和kd_acc对比
            print("[client{}, round{}] after pruning rate changed to {:.2f}, test acc: {:.2f}, kd acc: {:.2f}".format(
                self.id, self.round, self.cur_pruning_rate, acc, kd_acc))



        print("[client{}, round{}] finished training, acc: {:.2f}, time: {:.2f}, avg_loss: {:.6f}, entropy: {:.6f}, "
              "local_data_size: {}, pruning_rate: {:.2f}, training_intensity: {}, prune_time: {}".format(self.id,
                                                                                                         self.round,
                                                                                                         acc,
                                                                                                         total_time,
                                                                                                         avg_loss,
                                                                                                         entropy,
                                                                                                         local_data_size,
                                                                                                         self.cur_pruning_rate,
                                                                                                         self.training_intensity,
                                                                                                         prune_time))
        self.round += 1
        self.last_pruning_rate = self.cur_pruning_rate
        return acc, total_time, entropy, local_data_size, self.id, self.cur_pruning_rate, self.training_intensity, avg_loss, total_time + prune_time

    def _build_model(self, batch_norm=True):
        if self.model_name == 'MiniVGG':
            model = MiniVGG(dataset=self.dataset_name, batch_norm=batch_norm)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
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
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=True)
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
        # acc = self.local_test(model)
        # print(f"[client{self.id} accuracy] accuracy: {acc:.4f}")

        self.model = model
        # ✅ 返回 acc、训练时间、平均loss、信息熵、本地数据量Di
        return 0., total_time, avg_loss, entropy, local_data_size


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
        if pruning_rate == 0.:
            return model
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

    def finetune(self):
        # 训练少轮次，更新bn参数供剪枝算法使用
        model = self.model
        epochs = self.finetune_epochs
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []  # ✅ 用于存储所有batch的loss

        for epoch in range(epochs):
            if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            # training
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=True)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())  # ✅ 记录每个batch的loss
                loss.backward()
                optimizer.step()
                train_loader_tqdm.set_description(f'Train For Prune Epoch: {epoch} Loss: {loss.item():.6f}')

        self.model = model


    def fill_to_full_model_and_train(self):
        model = self.model
        full_model = self._build_model(self.batch_norm)
        mask = model.mask
        layer_idx_in_mask = 0
        if self.dataset_name == 'MNIST':
            start_mask = torch.ones(1).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
        else:
            start_mask = torch.ones(3).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
        end_mask = torch.tensor(mask[layer_idx_in_mask], dtype=torch.int).bool()
        if torch.cuda.is_available():
            model = model.to(self.device)
            full_model = full_model.to(self.device)
            start_mask = start_mask.to(self.device)
            end_mask = end_mask.to(self.device)
        for from_layer, to_layer in zip(model.modules(), full_model.modules()):
            start_indices = [i for i, x in enumerate(start_mask) if x]
            end_indices = [i for i, x in enumerate(end_mask) if x]
            if isinstance(from_layer, nn.Conv2d):
                with torch.no_grad():
                    for i, start_idx in enumerate(start_indices):
                        to_layer.weight.data[end_indices, start_idx, :, :] = from_layer.weight.data[:, i, :, :]
            if isinstance(from_layer, nn.BatchNorm2d):
                with torch.no_grad():
                    to_layer.weight.data[end_indices] = from_layer.weight.data
                    to_layer.bias.data[end_indices] = from_layer.bias.data
                    to_layer.running_mean.data[end_indices] = from_layer.running_mean.data
                    to_layer.running_var.data[end_indices] = from_layer.running_var.data
                layer_idx_in_mask += 1
                start_mask = end_mask[:]
                if layer_idx_in_mask < len(mask):
                    end_mask = mask[layer_idx_in_mask]
            if isinstance(from_layer, nn.Linear):
                with torch.no_grad():
                    for i, start_idx in enumerate(start_indices):
                        to_layer.weight.data[end_indices, start_idx] = from_layer.weight.data[:, i]
                    to_layer.bias.data[end_indices] = from_layer.bias.data
                layer_idx_in_mask += 1
                start_mask = end_mask[:]
                if layer_idx_in_mask < len(mask):
                    end_mask = mask[layer_idx_in_mask]
        epochs = self.training_epochs_for_prune  # for剪枝训练强度

        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(full_model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []  # ✅ 用于存储所有batch的loss
        s = self.s

        for epoch in range(epochs):
            # training
            full_model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=True)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = full_model(data)
                loss = criterion(output, target)
                losses.append(loss.item())  # ✅ 记录每个batch的loss

                loss.backward()
                for m in full_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1
                optimizer.step()
                train_loader_tqdm.set_description(f'Train For Prune Epoch: {epoch} Loss: {loss.item():.6f}')

        return full_model

    def distill_train(self, student_model, teacher_model, distill_epoch=10, T=2.0, alpha=0.7):
        """
        空模型（或随机初始化后）用全量 server_model 作为 teacher 做蒸馏。
        student_model: 已经按目标剪枝率处理好的学生模型
        teacher_model: 未剪枝的 server full 模型
        """
        student = student_model.to(self.device).train()
        teacher = teacher_model.to(self.device).eval()

        lr = (getattr(self, "optimizer_cfg", {}) or {}).get("lr", 0.01)
        momentum = (getattr(self, "optimizer_cfg", {}) or {}).get("momentum", 0.9)
        wd = (getattr(self, "optimizer_cfg", {}) or {}).get("wd", 0.0)
        opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        ce = torch.nn.CrossEntropyLoss()
        kl = torch.nn.KLDivLoss(reduction="batchmean")
        train_loader = self.load_train_data()

        for _ in range(int(distill_epoch)):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    t_logits = teacher(x)

                s_logits = student(x)

                # 硬标签 CE
                loss_ce = ce(s_logits, y)
                # 软标签 KD（注意温度平方系数）
                loss_kd = kl(
                    torch.nn.functional.log_softmax(s_logits / T, dim=1),
                    torch.nn.functional.softmax(t_logits / T, dim=1)
                ) * (T * T)

                loss = alpha * loss_kd + (1.0 - alpha) * loss_ce

                opt.zero_grad()
                loss.backward()
                opt.step()




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


    def for_kd_test(self, student_model):
        test_loader = self.load_test_data(batch_size=128)
        student_model = student_model.to(self.device)
        student_model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                data, target = data, target
            output = student_model(data)
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

    def fedavg_do(self):
        # 训练少轮次，更新bn参数供剪枝算法使用
        model = self.model
        epochs = self.training_intensity
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []  # ✅ 用于存储所有batch的loss
        start_time = time.time()
        for epoch in range(epochs):
            if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            # training
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, disable=True)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())  # ✅ 记录每个batch的loss

                loss.backward()
                optimizer.step()
                train_loader_tqdm.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')
        self.model = model
        end_time = time.time()
        total_time = end_time - start_time
        total = self.mock_time_delay(total_time)  # 模拟终端性能差异
        self.client_do_times.append(total)
        return total


    def fedprox_do(self, server_state_dict, epochs=None, mu=0.01, lr=0.1, momentum=0.9, weight_decay=0.0):
        """
        FedProx 本地训练：min_w  L_i(w) + (μ/2) * ||w - w_t||^2
        - server_state_dict: 服务端当前模型参数（state_dict()）
        - epochs: 本地训练轮数（若 None 则用 self.training_intensity，和你现有字段对齐）
        - mu: FedProx 的 μ 系数
        返回与 feddra_do 相同的 9 元组，便于 server 侧复用相同的收集/聚合逻辑。
        """
        start_all = time.time()

        model = self.model.to(self.device)
        model.train()

        # 参照你 fedavg_do 的写法：如果你有统一的 get_optimizer，就换成那个
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        local_epochs = int(epochs if epochs is not None else max(1, int(getattr(self, "training_intensity", 1))))
        train_loader = self.load_train_data()

        # 把 server 参数搬到本地 device，做一次快照（无梯度）
        server_params = {k: v.detach().to(self.device) for k, v in server_state_dict.items() if k in model.state_dict()}

        loss_meter, step_cnt = 0.0, 0
        tic = time.time()
        for ep in range(local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                ce = F.cross_entropy(logits, y)

                # FedProx proximal term： (μ/2) * Σ ||w - w_t||^2
                prox = 0.0
                for (name, p) in model.named_parameters():
                    if name in server_params and p.requires_grad:
                        prox = prox + (p - server_params[name]).pow(2).sum()

                loss = ce + 0.5 * mu * prox
                loss.backward()
                optimizer.step()

                loss_meter += ce.item()  # 记录纯分类损失，和你日志里“avg_loss”口径一致
                step_cnt += 1

        do_time = time.time() - tic
        avg_loss = loss_meter / max(step_cnt, 1)

        # 评估 + 指标
        local_data_size = int(getattr(self, "local_data_size", 0))
        total_time = time.time() - start_all
        # time mock一下
        # 为了与原有日志一致，这两个字段仍然回传
        client_last_pruning_rate = float(getattr(self, "cur_pruning_rate", 0.0))
        client_epochs = int(local_epochs)

        print(f"[client{self.id}, round{getattr(self, 'round_id', -1)}] "
              f"local_data_size: {local_data_size}, pruning_rate: {client_last_pruning_rate:.2f}, "
              f"training_intensity: {client_epochs}, prox_mu: {mu}, prune_time: 0.0")

        return (0, total_time, 0, local_data_size, int(self.id),
                client_last_pruning_rate, client_epochs, avg_loss, do_time)

    def _new_minivgg_student(self, sync_bn_from_server: bool = True, init: str = "kaiming"):
        """
        新建一个 MiniVGG 学生模型：
          - 只复制 BN 的 running_mean / running_var / num_batches_tracked（可选）
          - 其余权重随机初始化（默认 kaiming）
        """
        assert self.model_name == "MiniVGG", "Only MiniVGG is supported by _new_minivgg_student()."
        student = self._build_model(batch_norm=True).to(self.device).train()

        # 随机初始化权重（保持“空模型”公正性）
        if init == "kaiming":
            for m in student.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        # 可选：仅复制 BN 的 running stats，稳定训练，不影响“空权重”的设定
        if sync_bn_from_server:
            for sm, tm in zip(self.server_model.modules(), student.modules()):
                if isinstance(sm, torch.nn.BatchNorm2d) and isinstance(tm, torch.nn.BatchNorm2d):
                    tm.running_mean = sm.running_mean.detach().clone()
                    tm.running_var = sm.running_var.detach().clone()
                    tm.num_batches_tracked = sm.num_batches_tracked.detach().clone()

        return student

    def mock_time_delay(self, total_time):
        # 用于模拟终端之间的性能差异，通过训练时间来反馈
        if self.id % 4 == 1:  # 1 5 9
            total_time *= 2.
        elif self.id % 4 == 2:  # 2 6
            total_time *= 3.
        if self.id % 4 == 3:  # 3 7
            total_time *= 5.
        return total_time
