# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:08:56 2021

@author: mq20185996
"""

from imports import *
from util_funcs import *

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defaults.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    print(defaults.device)
    print(torch.cuda.device_count())

    seed = 30
    np.random.seed(seed)
    torch.manual_seed(seed)

    # stim_path = Path(r'D:\Andrea_NN\stimuli\no_transf')
    stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
    # stim_path = Path('../samediff_no-transf')
    epochs = 100
    cycles = 1
    batch_sz = 24
    lr_min = 1e-4
    weight_decay = 1e-3
    w_dropout_1 = 0.8
    w_dropout_2 = 0.8
    test_prop = 0.2

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    net_3 = SiameseNet12(w_dropout_1, w_dropout_2)
    net_4 = SiameseNet22(w_dropout_1, w_dropout_2)

    nets = [net_0, net_1, net_2, net_3, net_4]
    nets = [x.to(defaults.device) for x in nets]
    nets = [nn.DataParallel(x) for x in nets]

    criterion = nn.CrossEntropyLoss()

    train_networks(nets, criterion, stim_path, batch_sz, cycles, epochs,
                   lr_min, weight_decay, seed)
    test_noise(nets, criterion, stim_path, batch_sz, seed)
    test_fov_img(nets, criterion, stim_path, batch_sz, seed)
    test_classify(nets[1:], criterion, stim_path, batch_sz, seed)

# TODO:
# Try different feedback architectures
# - E.g., fully interconnected vs conv
# Deep net to fMRI mapping
