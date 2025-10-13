import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import read_client_data
from models.mini_vgg import MiniVGG
from PPO import DualStreamConfig, DualStreamPPO


class Server(object):
    def __init__(self, device, clients, dataset, memory_capacity,
                 d_glob=6, d_cli=7, p_low=0.2, p_high=0.9, E_min=1, E_max=5, hidden=256):
        self.device = device  # 'cpu' or 'cuda'
        self.clients = clients  # list of Client objects
        self.dataset = dataset  # string
        self.round_id = 0
        self.prev_acc = 0.0  # 上一轮全局模型精度
        self.memory_capacity = memory_capacity
        self.ppo_config = DualStreamConfig(d_glob=d_glob, d_cli=d_cli,
                                           p_low=p_low, p_high=p_high, E_min=E_min, E_max=E_max,
                                           hidden=hidden)

        self.agent = DualStreamPPO(self.ppo_config)
        self.ppo_update_every = 5
        self.init_models_save_path = './init_models/'
        self.server_model = MiniVGG(dataset=self.dataset)

    def do(self):
        self.generate_next_round_params()  # 收集上一轮FL指标
        self.aggregate()  # 聚合模型
        self.generate_client_model()  # 为client生成下一轮的模型


    def generate_client_model(self):
        """
        收集各client的剪枝率，生成每个client下一轮的模型
        """
        for client in self.clients:
            pass



    def generate_next_round_params(self):
        """
        Server 流程：
          A. 收集本轮各 client 的结果指标（这些指标对应“本轮动作”的效果）
          B. 构造 PPO 的 S_glob / S_cli / mask
          C. 写入 (state, action, reward) 到 PPO buffer
          D. 用策略生成“下一轮动作”并下发
          E. 到达间隔则 ppo_update()
        约定 client.do() 返回：
          acc, total_time, entropy, local_data_size, client_id, client_last_pruning_rate, client_epochs, avg_loss
        """

        # ===== A) 收集 client 指标（本轮已执行的动作及结果） =====
        times, entropies, sizes = [], [], []
        p_exec, E_exec, losses, ids, accs, prev_p = [], [], [], [], [], []

        for client in self.clients:
            acc, total_time, entropy, local_data_size, client_id, client_last_pruning_rate, client_epochs, avg_loss = client.do()
            times.append(float(total_time))
            entropies.append(float(entropy))
            sizes.append(int(local_data_size))
            p_exec.append(float(client_last_pruning_rate))  # 本轮“实际执行”的剪枝率
            E_exec.append(int(client_epochs))  # 本轮“实际执行”的本地轮数
            losses.append(float(avg_loss))
            ids.append(int(client_id))
            accs.append(float(acc))
            prev_p.append(float(client.pruning_rate))

        N = len(self.clients)
        if N == 0:
            return {}

        # ===== B) 构造 PPO 状态 =====
        # 全局特征 S_glob: [Acc, dAcc, Tmax, dT, round_id, bw]
        if hasattr(self, "evaluate_global") and callable(self.evaluate_global):
            acc_now = float(self.evaluate_global())
        else:
            acc_now = float(sum(accs) / max(len(accs), 1))  # 没有全局评估就用均值占位

        prev_acc = float(getattr(self, "prev_acc", 0.0))
        dAcc = acc_now - prev_acc
        Tmax = float(max(times)) if times else 0.0
        dT = float(max(times) - min(times)) if len(times) > 1 else 0.0
        bw = 1.0  # 可替换为真实带宽/拥塞指标

        S_glob = torch.tensor([[acc_now, dAcc, Tmax, dT, float(getattr(self, "round_id", 0)), bw]],
                              dtype=torch.float32, device=self.device)  # [1, 6]

        # 客户端特征 S_cli（与 DualStreamConfig.d_cli 对齐：7维）
        # [T_i, H_i, |D_i|, id_emb, p_prev, E_prev, loss_prev]
        rows = []
        for i in range(N):
            rows.append([
                times[i],
                entropies[i],
                sizes[i],
                float(ids[i]),  # 简单用 id 作“embedding”数值特征；也可换 learnable embedding
                p_exec[i],  # 本轮实际动作：剪枝率
                E_exec[i],  # 本轮实际动作：epochs
                losses[i],
            ])
        S_cli = torch.tensor([rows], dtype=torch.float32, device=self.device)  # [1, N, 7]
        mask = torch.ones(1, N, dtype=torch.float32, device=self.device)  # 全在线=1；掉线置0

        # ===== C) 写入经验 (state, action, reward) =====
        # 1) 构造全局奖励（示例：R = α·ΔAcc - β·Tmax - γ·ΔT - λ·CommCost）
        #    如果你有实际通信量，可据此计算 comm_cost；这里先置 0
        comm_cost = 0.0
        R = float(1.0 * dAcc - 0.01 * Tmax - 0.01 * dT - 0.0 * comm_cost)

        # 2) 动作张量（形状与 PPO 网络一致）
        p_tensor = torch.tensor([[[x] for x in p_exec]], dtype=torch.float32, device=self.device)  # [1, N, 1]
        E_tensor = torch.tensor([[[x] for x in E_exec]], dtype=torch.float32, device=self.device)  # [1, N, 1]

        # 3) 价值与 logp
        #    ✅ 最佳做法：在“上一次 select_actions 下发时”缓存每个 client 的 logp，取来用；
        #    ❗ 如果没缓存，这里用当前策略对“已执行动作”做 evaluate（近似 on-policy，亦可工作）
        eval_out = self.agent.net.evaluate_actions(S_glob, S_cli, p_tensor, E_tensor, mask)
        V_now = eval_out["V"].detach()  # [1, N, 1]
        # 如果你有缓存的 logp_old（上次 select 返回的），请用那个；否则用当前 evaluate 作为近似
        logp_now = eval_out["logp"].detach()  # [1, N]

        # 4) 写入 buffer
        self.agent.store_transition(dict(
            S_glob=S_glob, S_cli=S_cli, mask=mask,
            p=p_tensor, E=E_tensor,
            logp=logp_now,  # 若有缓存的 old_logp，请替换为 old_logp
            V=V_now,
            reward=torch.tensor([[R]], dtype=torch.float32, device=self.device),
            done=torch.zeros(1, 1, device=self.device),
        ))

        prev_p_tensor = torch.tensor([[[x] for x in prev_p]], dtype=torch.float32, device=self.device)
        # ===== D) 用策略为“下一轮”生成动作并下发 =====
        with torch.no_grad():
            out_next = self.agent.select_actions(S_glob, S_cli, mask, prev_p=prev_p_tensor)
        p_next = out_next["p"][0, :, 0].tolist()  # [N]
        E_next = [int(x) for x in out_next["E"][0, :, 0].tolist()]  # [N]

        for i, client in enumerate(self.clients):

            client.cur_pruning_rate = float(p_next[i])
            client.epochs = int(E_next[i])
            # 可选：把“下发时的 logp”缓存到 client 或 server 的某处，用于下一轮严格 on-policy
            client.last_action_logp = float(out_next["logp"][0, i].item())

        # 统计刷新（供下一轮 dAcc 使用）
        self.prev_acc = acc_now
        self.round_id = self.round_id + 1

        # ===== E) 到间隔则触发 PPO 更新 =====
        if len(self.agent.buffer.data) >= self.ppo_update_every:
            self.agent.ppo_update(epochs=4, batch_size=256, normalize_adv=True)
            # buffer 在 ppo_update 内部已清空
            print("PPO updated ✅")

        # 返回给上层（可选）
        return {c.id: dict(pruning_rate=float(p_next[i]), epochs=int(E_next[i]))
                for i, c in enumerate(self.clients)}

    def aggregate(self):
        server_model = self.server_model
        for param in server_model.parameters():
            param.data.zero_()  # 将簇模型参数都设置为0
        client_models = []

        for client in self.clients:
            client_model = client.load_model()
            client_mask = client_model.mask
            ratio = 1. / len(self.clients)
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


    def prune(self, model, pruning_rate):
        """从一个完整模型剪枝到剪枝率=pruning_rate模型"""
        model.generate_mask()
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
        new_model = MiniVGG(cfg=new_cfg, dataset=self.dataset).to(self.device)
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

