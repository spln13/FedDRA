import math
import torch
import torch.nn as nn


# MiniVGG的配置
mini_vgg_cfg = [32, 'M', 64, 'M', 128, 128]


class MiniVGG(nn.Module):
    def __init__(self, cfg=None, num_classes=10, dataset='cifar10', batch_norm=True, init_weights=True):
        super(MiniVGG, self).__init__()
        self.dataset = dataset
        self.mask = None
        if cfg is None:
            cfg = mini_vgg_cfg
        self.cfg = cfg
        self.features = self.make_layers(cfg, batch_norm)
        # 假设最后一个卷积层输出为 64 通道，我们进行两次 2x2 最大池化
        # 输入图像大小从 32x32 减少到 8x8
        self.classifier = nn.Linear(self.cfg[-1], num_classes)
        self._generate_mask()
        if init_weights:
            self.initialize_weights()

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        if self.dataset == 'MNIST' or self.dataset == 'emnist_noniid':
            in_channels = 1
        else:
            in_channels = 3  # CIFAR-10 是 RGB 三通道
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # v 是通道数量
                if i == len(cfg) - 1:
                    if self.dataset == "MNIST" or self.dataset == 'emnist_noniid':
                        conv2d = nn.Conv2d(in_channels, cfg[-1], kernel_size=7)  # mnist前一个层的输出是7x7
                    else:
                        conv2d = nn.Conv2d(in_channels, cfg[-1], kernel_size=8)  # 假设前一个层的输出是8x8
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平所有维度除了批次
        x = self.classifier(x)
        return x


    def _generate_mask(self):
        # 生成模型mask
        mask = []  # 初始mask生成全1
        # for item in cfg:
        #     if item == 'M':
        #         continue
        #     arr = [1.0 for _ in range(item)]
        #     mask.append(torch.tensor(arr))
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
                channels = module.weight.data.shape[0]
                arr = [1.0 for _ in range(channels)]
                mask.append(arr)
        self.mask = mask

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def fill_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


# 测试模型
if __name__ == "__main__":
    model = MiniVGG(num_classes=10)
    x = torch.randn(16, 3, 32, 32)  # CIFAR-10 输入大小为 32x32 RGB 图像
    output = model(x)
    print(output.shape)  # 应该输出 torch.Size([1, 10])
