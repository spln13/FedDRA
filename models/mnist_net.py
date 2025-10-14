import torch
import math
import torch.nn as nn
import torch.nn.functional as F


cnn_cfg = [16, 10]


class MNISTNet(nn.Module):
    def __init__(self, dataset='MNIST', cfg=None, num_classes=10, batch_norm=True, init_weights=True):
        super(MNISTNet, self).__init__()
        # Initial convolutional layer
        self.mask = None
        if cfg is None:
            cfg = cnn_cfg
        self.cfg = cfg
        self.name = 'mnistnet'
        self.dataset = dataset
        self.features = self.make_layers(cfg, batch_norm=batch_norm)
        self.classifier = nn.Linear(self.cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

        self._generate_mask()


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 10)
        x = self.classifier(x)
        return x


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # v 是通道数量
                if i == len(cfg) - 1:
                    conv2d = nn.Conv2d(in_channels, cfg[-1], kernel_size=28)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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
