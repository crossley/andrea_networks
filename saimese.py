# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:08:56 2021

@author: mq20185996
"""

from imports import *
from util_funcs import *

defaults.device = torch.device('cuda')

seed = 30
np.random.seed(seed)
torch.manual_seed(seed)

# stim_path = Path(r'D:\Andrea_NN\stimuli\no_transf')
stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
epochs = 50
cycles = 1
batch_sz = 24
lr_min = 1e-4
weight_decay = 1e-3
w_dropout_1 = 0.8
w_dropout_2 = 0.8

# NOTE: I use a different method to inject noise so always set to False
fov_noise = False

p = (stim_path, 
     epochs, 
     cycles, 
     batch_sz, 
     lr_min, 
     weight_decay, 
     w_dropout_1, 
     w_dropout_2, 
     fov_noise,
     seed)

# train_nets(p)
# test_nets_noise()

train_nets_2(p)
test_nets_noise_2()

# TODO: 
# Different feedback architectures
# - direct from V1
# - fully interconnected vs conv
# Deep net to fMRI mapping