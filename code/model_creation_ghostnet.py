# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
https://arxiv.org/abs/1911.11907
"""
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import math
import pandas as pd
import json
import geojson
import geopandas as gpd
import os.path
from datetime import datetime

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

__all__ = ['ghostnet']

'''custom torch dataset here'''


class readyData(Dataset):
    def __init__(self):
        all_ready_file = f"{github_dir}/data/ready_for_training/all_ready.csv"
        all_ready_pd = pd.read_csv(all_ready_file, header=0, index_col=0)
        print("read csv")
        all_ready_pd = all_ready_pd.fillna(10000)  # replace all nan with 10000
        all_ready_data = all_ready_pd[
            ['year', 'm', 'doy', 'ndsi', 'grd', 'eto', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',
             'vs', 'lat', 'lon', 'elevation', 'aspect', 'curvature', 'slope', 'eastness',
             'northness']].to_numpy().astype('float')
        all_ready_tgts = all_ready_pd['swe'].to_numpy().astype('float')

        all_ready_data = torch.from_numpy(all_ready_data)
        all_ready_tgts = torch.from_numpy(all_ready_tgts)

        self.data = []
        for i in range(0, len(all_ready_data)):
            temp = [all_ready_data[i], all_ready_tgts[i]]
            self.data.append(temp)

        print("dataset generation finished")
        #print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y


def my_collate(batch):
    dta = []
    target = []
    for item in batch:
        data_item = item[0].tolist()
        dta.append(data_item)
        target_item = item[1].tolist()
        target.append(target_item)

    dta = torch.FloatTensor(dta)
    dta = dta.unsqueeze(1)
    dta = dta.unsqueeze(2)
    # print(dta.shape)
    target = torch.FloatTensor(target)
    target = target.unsqueeze(1)
    # print(target.shape)
    # print(target)

    return dta, target


'''
train, test = train_test_split(all_ready_pd, test_size=0.2)
train_x, train_y = train[['year', 'm', 'doy', 'ndsi', 'grd', 'eto', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',
                                  'vs', 'lat', 'lon', 'elevation', 'aspect', 'curvature', 'slope', 'eastness',
                                  'northness']].to_numpy().astype('float'), train['swe'].to_numpy().astype('float')
test_x, test_y = test[['year', 'm', 'doy', 'ndsi', 'grd', 'eto', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',
                                 'vs', 'lat', 'lon', 'elevation', 'aspect', 'curvature', 'slope', 'eastness',
                                 'northness']].to_numpy().astype('float'), test['swe'].to_numpy().astype('float')
'''


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        print(output_channel)
        self.conv_stem = nn.Conv2d(1, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


split = 0.8
bs = 16
lr = 0.06
epochs = 100

if __name__ == '__main__':
    dset = readyData()
    train_set, test_set = random_split(dset,
                                       [math.floor(split * len(dset)), math.ceil((1 - split) * len(dset))])
    # load data and do things here
    trainLoader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=my_collate)
    testLoader = DataLoader(test_set, batch_size=bs, shuffle=True, collate_fn=my_collate)

    #prep model for training
    model = ghostnet()
    model.eval()
    # print(model)
    criterion = nn.L1Loss   ()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    #testing
    for epoch in range(epochs):

        # testing and training loss
        train_error = 0
        test_error = 0

        for i, data in enumerate(trainLoader):
            inputs, targets = data
            inputs = Variable(inputs, requires_grad=True)
            # print(inputs)
            # inputs = inputs.unsqueeze(0)
            # inputs = inputs.unsqueeze(1)
            # print(inputs.shape)
            targets = Variable(targets, requires_grad=True)

            optimizer.zero_grad()
            predict = model(inputs)
            # print(predict.shape)
            # print(predict)
            loss = criterion(predict, targets)
            train_error += loss.item() / len(trainLoader)

            loss.backward()
            optimizer.step()

        for i, data in enumerate(testLoader):
            inputs, targets = data
            # print(inputs)
            # print(targets)
            outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())
            test_error += loss.item() / (len(testLoader))

        # step lr scheduler
        # scheduler.step(test_error)

        print(f"Epoch {epoch + 1}, Train loss: {train_error}, Test loss: {test_error}")
        #train_errors.append(train_error)
        #test_errors.append(test_error)

        correct = 0
        total = 0

    with torch.no_grad():
        for i, data in enumerate(testLoader):
            inputs, targets = data
            outputs = model(inputs)
            #print(outputs)
            for (result, tgt) in zip(outputs, targets):
                if result == tgt:
                    correct += 1
                total+=1

    print('Accuracy of the network on the test set: %d out of %d' % (correct, total))
    print('Accuracy of the network on the test set: %d %%' % ((100 * correct / total)))


