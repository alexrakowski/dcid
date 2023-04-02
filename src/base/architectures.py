'''MobilenetV2 in PyTorch.

See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
from torchvision import models as tv_models
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            sample_size=224,
            width_mult=2,
            in_channels=3,
            last_channel=512,
            head_type='corr',
    ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = last_channel
        self.features = [conv_bn(in_channels, input_channel, (1, 2, 2))]
        self.head_type = head_type
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.features.append(nn.Flatten())
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# github.com/NYUMedML/CNN_design_for_AD/blob/master/models/models.py
class Widenet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            last_channel=512,
            width_mult=8,
            adaptive_pool=False,
            pool_type='max',
            num_classes=1,
            orig_impl=True,
    ):
        super().__init__()

        pool_cls, adaptive_pool_cls = (nn.MaxPool3d, nn.AdaptiveMaxPool3d) if pool_type == 'max' else (
            nn.AvgPool3d, nn.AdaptiveAvgPool3d)

        self.features = nn.Sequential()

        self.features.add_module('conv0_s1', nn.Conv3d(in_channels, 4 * width_mult, kernel_size=1))

        self.features.add_module('lrn0_s1', nn.InstanceNorm3d(4 * width_mult))
        self.features.add_module('relu0_s1', nn.ReLU(inplace=True))
        self.features.add_module('pool0_s1', pool_cls(kernel_size=3, stride=2))

        self.features.add_module('conv1_s1',
                                 nn.Conv3d(4 * width_mult, 32 * width_mult, kernel_size=3, padding=0, dilation=2))

        self.features.add_module('lrn1_s1', nn.InstanceNorm3d(32 * width_mult))
        self.features.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.features.add_module('pool1_s1', pool_cls(kernel_size=3, stride=2))

        self.features.add_module('conv2_s1',
                                 nn.Conv3d(32 * width_mult, 64 * width_mult, kernel_size=5, padding=2, dilation=2))

        self.features.add_module('lrn2_s1', nn.InstanceNorm3d(64 * width_mult))
        self.features.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.features.add_module('pool2_s1', pool_cls(kernel_size=3, stride=2))

        self.features.add_module('conv3_s1',
                                 nn.Conv3d(64 * width_mult, last_channel, kernel_size=3, padding=1, dilation=2))

        self.features.add_module('lrn3_s1', nn.InstanceNorm3d(last_channel))
        self.features.add_module('relu3_s1', nn.ReLU(inplace=True))

        if adaptive_pool:
            self.features.add_module('pool3_s1', adaptive_pool_cls((1, 1, 1)))
            self.features.add_module('flatten', nn.Flatten())
        else:
            if orig_impl:
                self.features.add_module('pool2_s1', pool_cls(kernel_size=5, stride=2))
                k = 5
            else:
                self.features.add_module('pool3_s1', pool_cls(kernel_size=5, stride=2))
                k = 1

            self.features.add_module('flatten', nn.Flatten())
            self.features.add_module('fc6_s1', nn.Linear(64 * width_mult * k ** 3, last_channel))

        self.fc = nn.Linear(last_channel, num_classes)


class DlibCNN(nn.Module):
    def __init__(
            self,
            in_channels,
            last_channel,
            dropout=0,
            batch_norm=False,
    ):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2),
                *([nn.BatchNorm2d(num_features=out_c)] if batch_norm else []),
                nn.ReLU(),
                nn.Dropout2d(p=dropout),
            )

        self.features = nn.Sequential(
            conv_block(in_c=in_channels, out_c=32),
            conv_block(in_c=32, out_c=32),
            conv_block(in_c=32, out_c=64),
            conv_block(in_c=64, out_c=64),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            *([nn.BatchNorm1d(num_features=256)] if batch_norm else []),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Linear(in_features=256, out_features=last_channel),
            *([nn.BatchNorm1d(num_features=last_channel)] if batch_norm else []),
            nn.Dropout1d(p=dropout),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.features(x)
        return features


class ResNet18(nn.Module):
    def __init__(
            self,
            last_channel,
            pretrained=False,
    ):
        super().__init__()

        features = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        expansion = features.layer1._modules['0'].expansion
        self.features = nn.Sequential(*list(features._modules.values())[:-1])

        self.features.append(nn.Flatten())
        if last_channel != 512 * expansion:
            self.features.append(nn.Linear(512 * expansion, last_channel))

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    model = MobileNetV2(num_classes=600, sample_size=112, width_mult=1)
    # model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 1, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
