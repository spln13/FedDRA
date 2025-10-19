# FedDRA: FedDRA: Reinforcement Learning-Driven Dynamic Resource Allocation for Efficient Federated Learning

## 思路以及细节
### 异构模型分配和聚合
1. client的每个模型都是由一个统一大模型剪枝得到 
2. 调用PPO得到每个模型的剪枝率，若剪枝率发生变化，从小模型拓展bn参数，重新训练再剪枝
3. 


## 应该做哪些实验
1. FedAvg
2. FedBN
3. FedProx
4. 再做一个强化学习的实验？
5. 消融实验（不使用PPO，使用纯策略模式）


## 实验应该关注哪些指标？
1. 训练总时长
2. 每一轮straggler等待时间
3. 准确率
4. PPO的reward

