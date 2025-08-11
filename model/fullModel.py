import torch
import torch.nn as nn
from torch.nn import Module
from prettytable import PrettyTable
import os
from .DCPBAM import DCPBAM
from .MFCAM import MFCAM
import sys

sys.path.append(os.path.dirname(__file__))


def conv_block(in_ch, out_ch):
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))
    return conv

def up_conv(in_ch, out_ch):
    up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return up

def classification_head(in_features, mid_features, out_features, dropout_rate):
        if mid_features is not None:
            r = nn.Sequential()
            r.add_module('linear_1', nn.Linear(in_features=in_features, out_features=mid_features))
            if dropout_rate is not None:
                if dropout_rate>0.0:
                    r.add_module('dropout', nn.Dropout(p=dropout_rate))
            r.add_module('relu_1', nn.ReLU())
            r.add_module('linear_2', nn.Linear(in_features=mid_features, out_features=out_features))
            return r
        else:
            return nn.Linear(in_features=in_features, out_features=out_features)

def conv_bn_acti_drop(
    in_channels,
    out_channels,
    kernel_size,
    activation=nn.ReLU,
    normalize=nn.BatchNorm2d,
    dropout_rate=0.0,
    sequential=None,
    name='',
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'):

    if sequential is None:
        r = nn.Sequential()
    else:
        r = sequential

    if len(name)==0:
        connector = ''
    else:
        connector = '_'

    r.add_module(
        name+connector+'conv2d',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode))

    if normalize is not None:
        norm_layer = normalize(out_channels)
        r.add_module(
            name+connector+norm_layer.__class__.__name__,
            norm_layer)

    if activation is not None:
        acti = activation()
        r.add_module(
            name+connector+acti.__class__.__name__,
            acti)

    if dropout_rate>0.0:
        r.add_module(
            name+connector+'dropout',
            nn.Dropout(p=dropout_rate))

    return r

class UNet_Chunk(Module):
    def __init__(self, in_channels, filter_list):
        super().__init__()

        self.in_channels = in_channels
        self.filter_list = filter_list

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(self.in_channels, self.filter_list[0])
        self.Conv2 = conv_block(self.filter_list[0], self.filter_list[1])
        self.Conv3 = conv_block(self.filter_list[1], self.filter_list[2])
        self.Conv4 = conv_block(self.filter_list[2], self.filter_list[3])
        self.Conv5 = conv_block(self.filter_list[3], self.filter_list[4])

        self.Dcpbam1 = DCPBAM(in_channels=self.filter_list[0], mid_channels=self.filter_list[0], cp_groups=4, spatial_kernel=1)
        self.Dcpbam2 = DCPBAM(in_channels=self.filter_list[1], mid_channels=self.filter_list[1], cp_groups=4, spatial_kernel=1)
        self.Dcpbam3 = DCPBAM(in_channels=self.filter_list[2], mid_channels=self.filter_list[2], cp_groups=4, spatial_kernel=1)
        self.Dcpbam4 = DCPBAM(in_channels=self.filter_list[3], mid_channels=self.filter_list[3], cp_groups=4, spatial_kernel=1)

        self.Up5 = up_conv(self.filter_list[4], self.filter_list[3])
        self.Up_conv5 = conv_block(self.filter_list[4], self.filter_list[3])

        self.Up4 = up_conv(self.filter_list[3], self.filter_list[2])
        self.Up_conv4 = conv_block(self.filter_list[3], self.filter_list[2])

        self.Up3 = up_conv(self.filter_list[2], self.filter_list[1])
        self.Up_conv3 = conv_block(self.filter_list[2], self.filter_list[1])

        self.Up2 = up_conv(self.filter_list[1], self.filter_list[0])
        self.Up_conv2 = conv_block(self.filter_list[1], self.filter_list[0])

    def forward(self, x):

        e1 = self.Conv1(x)
        e1_s = self.Dcpbam1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2_s = self.Dcpbam2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3_s = self.Dcpbam3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4_s = self.Dcpbam4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        d5 = torch.cat((e4_s, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3_s, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2_s, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1_s, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return e5, d2, d3, d4, d5


class FullModel(UNet_Chunk):

    def __init__(self, in_channels, filter_list, out_dict):
        super().__init__(in_channels, filter_list)
        self.out_dict = out_dict
        self.init()

        self.mfcam1 = MFCAM(in_channels=filter_list[-1])
        self.mfcam2 = MFCAM(in_channels=filter_list[0])
        self.mfcam3 = MFCAM(in_channels=filter_list[1])
        self.mfcam4 = MFCAM(in_channels=filter_list[2])
        self.mfcam5 = MFCAM(in_channels=filter_list[3])


    def init(self):
        self.dummy_tensor = nn.Parameter(torch.tensor(0), requires_grad=False)

        if self.out_dict is None:
            self.out_conv = nn.Conv2d(self.filter_list[0], 1, kernel_size=1, stride=1, padding=0)
        else:
            if 'class' in self.out_dict:
                if self.out_dict['class']>0:
                  # Cái này bản chất chỉ là 1 mạng neural
                    self.out_classification = classification_head(
                        in_features=self.filter_list[-1]+self.filter_list[0]+self.filter_list[1]+self.filter_list[2]+self.filter_list[3],
                        mid_features=self.filter_list[2],
                        out_features=self.out_dict['class'],
                        dropout_rate=0.25)
            if 'image' in self.out_dict:
                if self.out_dict['image']>0:
                    self.out_conv_image = conv_bn_acti_drop(
                        in_channels=self.filter_list[0],
                        out_channels=self.filter_list[0],
                        kernel_size=3,
                        activation=nn.ReLU,
                        normalize=nn.BatchNorm2d,
                        padding=1,
                        dropout_rate=0.0,
                        sequential=None)
                    self.out_conv_image.add_module(
                        'conv_last', nn.Conv2d(self.filter_list[0], self.out_dict['image'], kernel_size=1, stride=1, padding=0))


    def forward(self, x):
        e5, d2, d3, d4, d5 = super().forward(x)

        e5 = self.mfcam1(e5)
        d2 = self.mfcam2(d2)
        d3 = self.mfcam3(d3)
        d4 = self.mfcam4(d4)
        d5 = self.mfcam5(d5)

        r = []

        average_pool_e5 = e5.mean(dim=(-2,-1))
        average_pool_d2 = d2.mean(dim=(-2,-1))
        average_pool_d3 = d3.mean(dim=(-2,-1))
        average_pool_d4 = d4.mean(dim=(-2,-1))
        average_pool_d5 = d5.mean(dim=(-2,-1))

        average_pool = torch.cat((average_pool_e5, average_pool_d2, average_pool_d3, average_pool_d4, average_pool_d5), dim=1)
        y_class = self.out_classification(average_pool)

        y_class = torch.sigmoid(y_class)
        r.append(y_class)

        y_image = self.out_conv_image(d2)
        y_image = torch.sigmoid(y_image)
        r.append(y_image)

        return tuple(r)


# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params


# in_channels = 3
# base = 16
# filter_list = [base*(2**i) for i in range(5)]

# out_dict = {'class':3, 'image':1}

# model = FullModel(in_channels, filter_list, out_dict)

# count_parameters(model)


# img = torch.randn(2, 3, 256, 256)

# a, b = model(img)

# print(a)

# print(b.max())