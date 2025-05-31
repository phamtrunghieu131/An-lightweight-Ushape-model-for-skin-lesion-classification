import torch
import torch.nn as nn
from torch.nn import Module


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

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return e5, d2

class FullModel(UNet_Chunk):

    def __init__(self, in_channels, filter_list, out_dict):
        super().__init__(in_channels, filter_list)
        self.out_dict = out_dict
        self.init()

    def init(self):

        self.out_classification = classification_head(
            in_features=self.filter_list[0]+self.filter_list[-1],
            mid_features=self.filter_list[0],
            out_features=self.out_dict['class'],
            dropout_rate=0.25)


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
        e5, d2 = super().forward(x)

        r = []

        average_pool_e5 = e5.mean(dim=(-2,-1))
        average_pool_d2 = d2.mean(dim=(-2,-1))
        average_pool = torch.cat((average_pool_e5, average_pool_d2), dim=1)
        y_class = self.out_classification(average_pool)
        y_class = torch.sigmoid(y_class)
        r.append(y_class)


        y_image = self.out_conv_image(d2)
        y_image = torch.sigmoid(y_image)
        r.append(y_image)


        return tuple(r)