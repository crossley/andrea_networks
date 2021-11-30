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
    stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
    # stim_path = Path('../samediff_no-transf')
    epochs = 10
    cycles = 1
    batch_sz = 24
    lr_min = 1e-4
    weight_decay = 1e-3
    w_dropout_1 = 0.8
    w_dropout_2 = 0.8

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    net_3 = SiameseNet12(w_dropout_1, w_dropout_2)
    net_4 = SiameseNet22(w_dropout_1, w_dropout_2)

    nets = [net_0, net_1, net_2, net_3, net_4]

    for net in nets:
        net.init_weights()
        net.init_pretrained_weights()
        net.freeze_pretrained_weights()
        params_to_update = net.parameters()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      params_to_update),
                               lr=lr_min,
                               weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        data_loader = make_dls(stim_path, batch_sz, seed)
        net.train_net(optimizer, criterion, data_loader, cycles, epochs)
        torch.save(net.state_dict(), '../trained_nets/net_' + net.module.model_name + '.pth')

    # res = test_nets_noise(p)
    # inspect_results_test(res)

    # res = test_nets_fovimg(p)
    # res = pd.concat(res)
    # print(res)
    # TODO: need to implement this
    # inspect_results_test_fovimg(res)

    # res = test_nets_fov_decode(p)
    # TODO: need to implement this
    # inspect_results_test_fov_decode(res)
