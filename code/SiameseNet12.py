# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *


class SiameseNet12(SiameseNet):
    def __init__(self, w_dropout_1, w_dropout_2):

        super(SiameseNet12, self).__init__(w_dropout_1, w_dropout_2, 2)

        self.model_name = 'SiameseNet12'

        self.head_mult = 2

        # self.fb = nn.Sequential(
        #     nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=197),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )

        self.fb = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(256,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Linear(in_features=256,
                      out_features=3,
                      bias=False),
        )

    def forward(self, inp):

        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        v1_p1 = self.V1(inp1)
        v2_p1 = self.V2(v1_p1)
        v4_p1 = self.V4(v2_p1)
        vIT_p1 = self.IT(v4_p1)

        v1_p2 = self.V1(inp2)
        v2_p2 = self.V2(v1_p2)
        v4_p2 = self.V4(v2_p2)
        vIT_p2 = self.IT(v4_p2)

        # feedback from v1
        p_cat = torch.cat((v1_p1, v1_p2), 1)
        fb = self.fb(p_cat)

        v1_fov = self.V1(fb + fov_inp)
        v2_fov = self.V2(v1_fov)
        v4_fov = self.V4(v2_fov)
        vIT_fov = self.IT(v4_fov)

        out = torch.cat((vIT_p1 + vIT_fov, vIT_p2 + vIT_fov), 1)

        out = self.head(out)

        return out
