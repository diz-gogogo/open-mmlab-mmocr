import torch
from torch import nn
from torch.nn import functional as F
import numpy


# 扩大两倍的上采样
class ConvTranspose2dModule(nn.Module):
    def __init__(self, in_channel):
        super(ConvTranspose2dModule, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channel // 2, in_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 2)
        )

    def forward(self, X):
        out = self.conv_trans(X)
        return out


# 扩大两倍的上采样
class Conv2dBNModule(nn.Module):
    def __init__(self, in_channel):
        super(Conv2dBNModule, self).__init__()

        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(in_channel * 2)
        )

    def forward(self, X):
        # print(X.size(1))
        out = self.conv_bn(X)
        return out


class ANet(nn.Module):
    def __init__(self, in_channel):
        super(ANet, self).__init__()

        self.in_channels = in_channel

        self.up_conv1 = ConvTranspose2dModule(self.in_channels)

        self.up_conv2 = ConvTranspose2dModule(self.in_channels // 2)

        self.up_conv3 = ConvTranspose2dModule(self.in_channels // 4)

        self.down_conv3 = Conv2dBNModule(self.in_channels // 8)

        self.down_conv2 = Conv2dBNModule(self.in_channels // 4)

        self.down_conv1 = Conv2dBNModule(self.in_channels // 2)

    def forward(self, X):
        out_up_conv1 = self.up_conv1(X)
        out_up_conv2 = self.up_conv2(out_up_conv1)
        out_up_conv3 = self.up_conv3(out_up_conv2)

        # print("up_1", out_up_conv1.shape)
        # print("up_2", out_up_conv2.shape)
        # print("up_3", out_up_conv3.shape)

        out_conv3_down = self.down_conv3(out_up_conv3)
        out_conv2_down = self.down_conv2(out_conv3_down)
        out_conv1_down = self.down_conv1(out_conv2_down)

        # print("down_1", out_conv3_down.shape)
        # print("down_2", out_conv2_down.shape)
        # print("down_3", out_conv1_down.shape)

        out = torch.concat((X, out_conv1_down), dim=1)

        return out


def mian():
    temp_X = torch.randn([1, 16, 16, 160])

    input_channel = temp_X.size(1)

    a_net = ANet(input_channel)

    out = a_net(temp_X)

    # print(out.shape)


if __name__ == '__main__':
    # mian()
    pass
