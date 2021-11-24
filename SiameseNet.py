# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *

class SiameseNet(nn.Module):
    def __init__(self, w_dropout_1, w_dropout_2, forward_ind):
        super(SiameseNet, self).__init__()
        
        self.forward_ind = forward_ind

        # V1 layers
        self.V1_p = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V2 layers
        self.V2_p = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V4 layers
        self.V4_p = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # IT layers
        self.IT_p = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V1 layers
        self.V1_f = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2,
                      padding=7 // 2),  # + self.vfb,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V2 layers
        self.V2_f = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V4 layers
        self.V4_f = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # IT layers
        self.IT_f = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # feedback from periphery to fovea: IT to v1
        self.fb = nn.Sequential(
            nn.Conv2d(1024, 3, kernel_size=3, stride=1, padding=221),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # feedback from periphery to fovea: v1 to v1
        # TODO: padding probably needs fixing
        self.fb2 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=197),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # head
        if forward_ind == 1:
            batch_mult = 2
        elif forward_ind == 2:
            batch_mult = 3
        elif forward_ind == 3:
            batch_mult = 2
        elif forward_ind == 12:
            batch_mult = 2
        elif forward_ind == 22:
            batch_mult = 3
        elif forward_ind == 32:
            batch_mult = 2
            
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(1024 * batch_mult,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Dropout(p=w_dropout_1, inplace=False),
            nn.Linear(in_features=1024 * batch_mult, out_features=512, bias=False),
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
        
        if self.forward_ind == 1:
            out = self.forward_1(inp)
            
        if self.forward_ind == 2:
            out = self.forward_2(inp)
            
        if self.forward_ind == 3:
            out = self.forward_3(inp)
            
        if self.forward_ind == 12:
            out = self.forward_12(inp)
            
        if self.forward_ind == 22:
            out = self.forward_22(inp)
            
        if self.forward_ind == 32:
            out = self.forward_32(inp)
            
        return out

    def forward_1(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((vIT_p1, vIT_p2), 1)

        fb = self.fb(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1, vIT_p2), 1)
    
        out = self.head(out_all)
        
        return out
        
    def forward_2(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((vIT_p1, vIT_p2), 1)

        fb = self.fb(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1, vIT_p2, vIT_fov), 1)
    
        out = self.head(out_all)

        return out
    
    def forward_3(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((vIT_p1, vIT_p2), 1)

        fb = self.fb(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1 + vIT_fov, vIT_p2 + vIT_fov), 1)
    
        out = self.head(out_all)
        
    def forward_12(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((v1_p1, v1_p2), 1)

        fb = self.fb2(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1, vIT_p2), 1)
    
        out = self.head(out_all)
        
        return out
        
    def forward_22(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((v1_p1, v1_p2), 1)

        fb = self.fb2(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1, vIT_p2, vIT_fov), 1)
    
        out = self.head(out_all)

        return out
    
    def forward_32(self, inp):
        
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1_p(inp1)
        v2_p1 = self.V2_p(v1_p1)
        v4_p1 = self.V4_p(v2_p1)
        vIT_p1 = self.IT_p(v4_p1)

        # perihperal 1
        v1_p2 = self.V1_p(inp2)
        v2_p2 = self.V2_p(v1_p2)
        v4_p2 = self.V4_p(v2_p2)
        vIT_p2 = self.IT_p(v4_p2)

        out_cat = torch.cat((v1_p1, v1_p2), 1)

        fb = self.fb2(out_cat)
        
        # fovea
        v1_fov = self.V1_f(fb + fov_inp)
        v2_fov = self.V2_f(v1_fov)
        v4_fov = self.V4_f(v2_fov)
        vIT_fov = self.IT_f(v4_fov)

        out_all = torch.cat((vIT_p1 + vIT_fov, vIT_p2 + vIT_fov), 1)
    
        out = self.head(out_all)

        return out