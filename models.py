# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:36:13 2023

@author: QNK
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from utils import CReLU


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        bias = True
        self.conv1 = nn.Conv2d(1, 4, (2, 2), stride=2, bias=bias, dtype=torch.cfloat)
        self.convx = nn.Conv2d(4, 4, (2, 2), bias=bias, dtype=torch.cfloat)
        self.convy = nn.Conv2d(4, 4, (2, 2), bias=bias, dtype=torch.cfloat)
        self.convz = nn.Conv2d(4, 4, (2, 2), bias=bias, dtype=torch.cfloat)
        self.convi = nn.Conv2d(4, 4, (2, 2), bias=bias, dtype=torch.cfloat)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(16, 64)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(64, 64)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(64, 32)),
                    ("relu3", nn.ReLU()),
                    ("fc4", nn.Linear(32, 1)),
                    # ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def round_sigmoid(self, y):
        return torch.round(F.sigmoid(y))

    def forward(self, x):
        x_conv1 = self.conv1(x)

        x_convx = self.convx(x_conv1)
        x_convy = self.convx(x_conv1)
        x_convz = self.convx(x_conv1)
        x_convi = self.convx(x_conv1)

        x16 = torch.cat((x_convx, x_convy, x_convz, x_convi), dim=1).squeeze()

        y_hat = self.fc(torch.abs(x16))

        # if not self.training:
        #     y_hat = self.sigmoid(y_hat)

        return y_hat.squeeze()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        bias = False
        dtype = torch.cfloat

        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 8, (2, 2), bias=bias, dtype=dtype)),
                    # ("bn1", nn.BatchNorm2d(4, dtype=dtype)),
                    #("relu1", CReLU()),
                    # ("do1", nn.Dropout2d(p=0.25)),
                    ("convx", nn.Conv2d(8, 16, (2, 2), bias=bias, dtype=dtype)),
                    # ("bn2", nn.BatchNorm2d(8, dtype=dtype)),
                    #("relu2", CReLU()),
                    # ("do2", nn.Dropout2d(p=0.25)),
                    #("convy", nn.Conv2d(128, 256, (2, 2), bias=bias, dtype=dtype)),
                    # ("bn3", nn.BatchNorm2d(16, dtype=dtype)),
                    #("relu3", CReLU()),
                    # ("do3", nn.Dropout2d(p=0.25)),
                    # ("convi", nn.Conv2d(4, 4, (2, 2), bias=bias, dtype=dtype)),
                    # ("bn3", nn.BatchNorm2d(4, dtype=dtype)),
                    # ("relu3", nn.ReLU()),
                ]
            )
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(16, 1024)),
                    ("bn1", nn.BatchNorm1d(1024)),
                    ("relu1", nn.ReLU()),
                    ("do1", nn.Dropout(p=0.2)),
                    ("fc2", nn.Linear(1024, 512)),
                    ("bn2", nn.BatchNorm1d(512)),
                    ("relu2", nn.ReLU()),
                    ("do2", nn.Dropout(p=0.2)),
                    ("fc3", nn.Linear(512, 256)),
                    ("bn3", nn.BatchNorm1d(256)),
                    ("relu3", nn.ReLU()),
                    ("do3", nn.Dropout(p=0.2)),
                    ("fc4", nn.Linear(256, 1)),
                ]
            )
        )

    def round_sigmoid(self, y):
        return torch.round(F.sigmoid(y))

    def forward(self, x):
        x_conv = self.conv(x)
        y_hat = self.fc(torch.abs(x_conv.squeeze()))
        return y_hat.squeeze()


class Branching(nn.Module):
    def __init__(self, branch: int = 6):
        super().__init__()

        if not 1 <= branch <= 15:
            raise Exception("branch must be between 1 and 15")

        self.branch = branch
        bias = False
        dtype = torch.cfloat
        # p = 0.2
        
        self.conv1 = nn.Conv2d(1, 1, (2, 2), stride=2, bias=bias, dtype=dtype)
        self.conv2 = nn.Conv2d(1, 1, (2, 2), stride=2, bias=bias, dtype=dtype)
        self.conv3 = nn.Conv2d(1, 1, (2, 2), stride=2, bias=bias, dtype=dtype)
        self.conv4 = nn.Conv2d(1, 1, (2, 2), stride=2, bias=bias, dtype=dtype)

        self.convx = nn.Conv2d(1, 1, (2, 2), bias=bias, dtype=dtype)
        self.convy = nn.Conv2d(1, 1, (2, 2), bias=bias, dtype=dtype)
        self.convz = nn.Conv2d(1, 1, (2, 2), bias=bias, dtype=dtype)
        self.convi = nn.Conv2d(1, 1, (2, 2), bias=bias, dtype=dtype)

        self.layer1 = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.layer2 = [self.convx, self.convy, self.convz, self.convi]
        
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(self.branch, 1024)),
                    ("relu1", nn.ReLU()),
                    ("bn1", nn.BatchNorm1d(1024)),
                    # ("do1", nn.Dropout(p=p)),
                    ("fc2", nn.Linear(1024, 512)),
                    ("relu2", nn.ReLU()),
                    ("bn2", nn.BatchNorm1d(512)),
                    # ("do2", nn.Dropout(p=p)),
                    ("fc3", nn.Linear(512, 256)),
                    ("relu3", nn.ReLU()),
                    ("bn3", nn.BatchNorm1d(256)),
                    # ("do3", nn.Dropout(p=p)),
                    ("fc4", nn.Linear(256, 1)),
                ]
            )
        )

    def round_sigmoid(self, y):
        return torch.round(F.sigmoid(y))

    def forward(self, x):
        b = 0
        sigma_kernel = []
        for k1 in self.layer1:
            x1 = k1(x)
            for k2 in self.layer2:
                sigma_kernel.append(k2(x1))
                b += 1
                if b == self.branch:
                    break
            else:
                continue
            break

        x_branched = torch.cat(sigma_kernel, dim=1).squeeze().reshape((-1, self.branch))
        y_hat = self.fc(x_branched.real)

        return y_hat.squeeze()


class Linear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dim, 15)),
                    ("fc2", nn.Linear(15, 1024)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(1024, 1024)),
                    ("relu3", nn.ReLU()),
                    ("fc4", nn.Linear(1024, 512)),
                    ("relu4", nn.ReLU()),
                    ("fc5", nn.Linear(512, 256)),
                    ("relu5", nn.ReLU()),
                    ("fc6", nn.Linear(256, 128)),
                    ("relu6", nn.ReLU()),
                    ("fc7", nn.Linear(128, 1)),
                ]
            )
        )

    def round_sigmoid(self, y):
        return torch.round(F.sigmoid(y))

    def forward(self, x):
        return self.fc(x)
