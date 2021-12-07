# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *


class SiameseNet0(SiameseNet):
    def __init__(self, w_dropout_1, w_dropout_2):

        super(SiameseNet0, self).__init__(w_dropout_1, w_dropout_2, 3)

        self.model_name = 'SiameseNet0'

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

        # no feedback from periphery to fov in this model
        v1_fov = self.V1(fov_inp)
        v2_fov = self.V2(v1_fov)
        v4_fov = self.V4(v2_fov)
        vIT_fov = self.IT(v4_fov)

        out = torch.cat((vIT_p1, vIT_p2, vIT_fov), 1)

        out = self.head(out)

        return out
