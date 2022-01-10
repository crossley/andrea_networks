# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *


class SiameseNet13(SiameseNet):
    def __init__(self, w_dropout_1, w_dropout_2):

        super(SiameseNet13, self).__init__(w_dropout_1, w_dropout_2, 3)

        self.model_name = 'SiameseNet13'

        self.fb = nn.Sequential(
            nn.Conv2d(128, 61, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
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

        p_cat = torch.cat((v1_p1, v1_p2), 1)
        fb = self.fb(p_cat)

        m = nn.Upsample((fov_inp.size()[2], fov_inp.size()[3]),
                        mode='bilinear')
        fb = m(fb)

        # x = fb.cpu().detach().numpy()
        # print(x.shape)
        # nrow=5
        # ncol=5
        # fig, ax = plt.subplots(nrow, ncol, squeeze=False, figsize=(10, 10))
        # for i in range(nrow):
        #     for j in range(ncol):
        #         ax[i, j].imshow(x[0, i+j, :, :])
        # [a.set_xticks([]) for a in ax.flatten()]
        # [a.set_yticks([]) for a in ax.flatten()]
        # plt.subplots_adjust(hspace=0.0, wspace=0.0)
        # plt.show()

        # x = v1_p1.cpu().detach().numpy()
        # print(x.shape)
        # nrow=5
        # ncol=5
        # fig, ax = plt.subplots(nrow, ncol, squeeze=False, figsize=(10, 10))
        # for i in range(nrow):
        #     for j in range(ncol):
        #         ax[i, j].imshow(x[0, i+j, :, :])
        # [a.set_xticks([]) for a in ax.flatten()]
        # [a.set_yticks([]) for a in ax.flatten()]
        # plt.subplots_adjust(hspace=0.0, wspace=0.0)
        # plt.show()

        # x = v1_p2.cpu().detach().numpy()
        # print(x.shape)
        # nrow=5
        # ncol=5
        # fig, ax = plt.subplots(nrow, ncol, squeeze=False, figsize=(10, 10))
        # for i in range(nrow):
        #     for j in range(ncol):
        #         ax[i, j].imshow(x[0, i+j, :, :])
        # [a.set_xticks([]) for a in ax.flatten()]
        # [a.set_yticks([]) for a in ax.flatten()]
        # plt.subplots_adjust(hspace=0.0, wspace=0.0)
        # plt.show()

        v1_fov_input = torch.cat((fov_inp, fb), 1)

        v1_fov = self.V1_fov(v1_fov_input)
        v2_fov = self.V2(v1_fov)
        v4_fov = self.V4(v2_fov)
        vIT_fov = self.IT(v4_fov)

        out = torch.cat((vIT_p1, vIT_p2, vIT_fov), 1)
        # out = vIT_fov

        out = self.head(out)

        return out
