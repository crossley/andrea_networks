# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:08:56 2021

@author: mq20185996
"""

# TODO:
# Try different feedback architectures
# - fully interconnected vs conv
# Deep net to fMRI mapping

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
    # stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
    stim_path = Path('../samediff_no-transf')
    epochs = 100
    cycles = 1
    batch_sz = 24
    lr_min = 1e-4
    weight_decay = 1e-3
    w_dropout_1 = 0.8
    w_dropout_2 = 0.8

    p = (stim_path, epochs, cycles, batch_sz, lr_min, weight_decay,
         w_dropout_1, w_dropout_2, seed)

    # train_nets(p)

    # res = test_nets_noise(p)
    # inspect_results_test(res)

    res = test_nets_fovimg(p)
    # TODO: need to implement this
    # inspect_results_test_fovimg(res)

    # res = test_nets_fov_decode(p)
    # TODO: need to implement this
    # inspect_results_test_fov_decode(res)
