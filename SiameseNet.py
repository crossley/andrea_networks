# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *


class SiameseNet(nn.Module):
    def __init__(self, w_dropout_1, w_dropout_2, head_mult):
        super(SiameseNet, self).__init__()

        self.V1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.IT = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(1024 * head_mult,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Dropout(p=w_dropout_1, inplace=False),
            nn.Linear(in_features=1024 * head_mult,
                      out_features=512,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Dropout(p=w_dropout_2, inplace=False),
            nn.Linear(in_features=512, out_features=2, bias=False),
        )

    def forward(self, inp):
        pass
