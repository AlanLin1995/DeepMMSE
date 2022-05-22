import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util


class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, n_file, is_dropout=True, is_relu=False, is_sigmoid=False, p=0.075):
        super(conv_layer, self).__init__()
        self.is_dropout = is_dropout
        self.is_relu = is_relu
        self.is_simoid = is_sigmoid
        self.dropout = nn.Dropout(p=p)
        self.conv = nn.Conv2d(in_channels * n_file, out_channels * n_file, kernel_size=3, stride=1, groups=n_file,
                              padding=(1, 1))
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        #            pass
        x = self.conv(x)
        if self.is_simoid:
            x = self.sigmoid(x)
        else:
            x = self.lrelu(x)
        return x


def concat(x, y, n_file):
    lx = list(x.shape)
    ly = list(y.shape)
    h = min(lx[2], ly[2])
    w = min(lx[3], ly[3])
    x = x.view(lx[0], n_file, -1, lx[2], lx[3])
    y = y.view(ly[0], n_file, -1, ly[2], ly[3])
    x = torch.cat([x[:, :, :, 0:h, 0:w], y[:, :, :, 0:h, 0:w]], dim=2)
    x = x.view(lx[0], -1, h, w)
    return x


class network(nn.Module):
    def __init__(self, in_channels, n_file, num_mask, sigma):
        super(network, self).__init__()
        self.n_file = n_file

        mid_channels = round(24 * num_mask ** 0.5)
        out_channels_1 = round(32 * num_mask ** 0.5)
        out_channels_2 = round(16 * num_mask ** 0.5)
        #        p_list = [0.075, 0.075,
        #                  0.15,
        #                  0.225,
        #                  0.3,
        #                  0.375,
        #                  0.45,
        #                  0.375, 0.375,
        #                  0.3, 0.3,
        #                  0.225, 0.225,
        #                  0.15, 0.15,
        #                  0.075, 0.075, 0.075]
        #        p_list = [x+0.35 for x in p_list]
        #        p_list = [0.1, 0.1,
        #                  0.2,
        #                  0.3,
        #                  0.4,
        #                  0.5,
        #                  0.6,
        #                  0.5, 0.5,
        #                  0.4, 0.4,
        #                  0.3, 0.3,
        #                  0.2, 0.2,
        #                  0.1, 0.1, 0.1]
        #        p_list = [x+0.25 for x in p_list]
        #        p_list = [0.5, 0.5,
        #                  0.55,
        #                  0.6,
        #                  0.65,
        #                  0.7,
        #                  0.75,
        #                  0.7, 0.7,
        #                  0.65, 0.65,
        #                  0.6, 0.6,
        #                  0.55, 0.55,
        #                  0.5, 0.5, 0.5]
        #        p_list = [x-0.1 for x in p_list]

        if sigma == 9:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 for x in p_list]

        if sigma == 27:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 + 0.2 for x in p_list]

        if sigma == 81:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 + 0.4 for x in p_list]

        ########################################################

        if sigma == 10:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 + 0.2 for x in p_list]

        if sigma == 15:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 + 0.1 for x in p_list]

        if sigma == 20:
            p_list = [0.075, 0.075,
                      0.15,
                      0.225,
                      0.3,
                      0.375,
                      0.45,
                      0.375, 0.375,
                      0.3, 0.3,
                      0.225, 0.225,
                      0.15, 0.15,
                      0.075, 0.075, 0.075]
            p_list = [x + 0.025 for x in p_list]

        self.conv1 = conv_layer(in_channels, mid_channels, n_file, p=p_list[0])
        self.conv2 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[1])
        self.conv3 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[2])
        self.conv4 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[3])
        self.conv5 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[4])
        self.conv6 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[5])
        self.conv7 = conv_layer(mid_channels, mid_channels, n_file, p=p_list[6])

        self.conv8 = conv_layer(mid_channels * 2, mid_channels * 2, n_file, p=p_list[7])
        self.conv9 = conv_layer(mid_channels * 2, mid_channels * 2, n_file, p=p_list[8])
        self.conv10 = conv_layer(mid_channels * 3, mid_channels * 2, n_file, p=p_list[9])
        self.conv11 = conv_layer(mid_channels * 2, mid_channels * 2, n_file, p=p_list[10])
        self.conv12 = conv_layer(mid_channels * 3, mid_channels * 2, n_file, p=p_list[11])
        self.conv13 = conv_layer(mid_channels * 2, mid_channels * 2, n_file, p=p_list[12])
        self.conv14 = conv_layer(mid_channels * 3, mid_channels * 2, n_file, p=p_list[13])
        self.conv15 = conv_layer(mid_channels * 2, mid_channels * 2, n_file, p=p_list[14])
        self.conv16 = conv_layer(mid_channels * 2 + in_channels, out_channels_1, n_file, p=p_list[15])
        self.conv17 = conv_layer(out_channels_1, out_channels_2, n_file, p=p_list[16])
        self.conv18 = conv_layer(out_channels_2, in_channels, n_file, p=p_list[17], is_sigmoid=True)

    def forward(self, x, is_dropout=True):

        x = x + torch.randn(x.shape).to(0) * (1. / 30)  # 256
        x = x
        skips = [x]

        # Encoder-----------------------------------------------
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)  # 128
        skips.append(x)

        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)  # 64
        skips.append(x)

        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)  # 32
        skips.append(x)

        x = self.conv5(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)  # 16
        skips.append(x)

        x = self.conv6(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)  # 8
        x = self.conv7(x)

        # Dncoder-----------------------------------------------
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)  # 16
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv8(x)
        x = self.conv9(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)  # 32
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv10(x)
        x = self.conv11(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)  # 64
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv12(x)
        x = self.conv13(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)  # 128
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv14(x)
        x = self.conv15(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)  # 256
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

        return x
