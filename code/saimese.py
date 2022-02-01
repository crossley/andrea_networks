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

    stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
    # stim_path = Path('../samediff_no-transf')
    # stim_path = Path('../abstract_stimuli/')
    # stim_path = Path('../samediff_no-transf_tiny')
    stim_path_abstract = Path('../abstract_stimuli')

    epochs = 50
    cycles = 1
    batch_sz = 24
    lr_min = 1e-6
    weight_decay = 1e-3
    w_dropout_1 = 0.2
    w_dropout_2 = 0.2
    test_prop = 0.2

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    # net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    # net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    # net_02 = SiameseNet02(w_dropout_1, w_dropout_2)
    # net_12 = SiameseNet12(w_dropout_1, w_dropout_2)
    # net_22 = SiameseNet22(w_dropout_1, w_dropout_2)
    net_13 = SiameseNet13(w_dropout_1, w_dropout_2)
    net_23 = SiameseNet23(w_dropout_1, w_dropout_2)

    nets = [net_23, net_13]
    nets = [x.to(defaults.device) for x in nets]
    nets = [nn.DataParallel(x) for x in nets]

    # for net in nets:
    #     net.module.init_weights()
    #     net.module.init_pretrained_weights()
    #     net.module.freeze_pretrained_weights()
    #     params_to_update = net.parameters()

    # print(summary(nets[2], input_size=(3, batch_sz, 3, 224, 224)))

    # net_13.to(defaults.device)
    # net_13 = nn.DataParallel(net_13)
    # net_23.to(defaults.device)
    # net_23 = nn.DataParallel(net_23)

    # criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyLossFlat()

    # NOTE: fmri stim training / testing
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # train_networks(nets, criterion, dls, batch_sz, cycles, epochs, lr_min,
    #                 weight_decay, seed, 'real_stim')
    # test_fov_img(nets, criterion, stim_path, batch_sz, seed, 'real_stim')
    # test_noise(nets, criterion, stim_path, batch_sz, seed, 'real_stim')
    # test_classify(nets, criterion, stim_path, batch_sz, seed, 'real_stim')

    # NOTE: abstract stim training / testing
    # dls = make_dls_abstract(stim_path_abstract, get_img_tuple_fov_empty_abstract,
    #                         batch_sz, seed)
    # train_networks(nets, criterion, dls, batch_sz, cycles, epochs, lr_min,
    #                 weight_decay, seed, 'abstract_stim')
    gc.collect()
    torch.cuda.empty_cache()
    test_noise(nets, criterion, stim_path_abstract, batch_sz, seed,
                'abstract_stim')
    # test_fov_img(nets, criterion, stim_path_abstract, batch_sz, seed,
    #              'abstract_stim')
    # test_classify(nets, criterion, stim_path_abstract, batch_sz, seed, 'abstract_stim')

    # inspect weights / features
    # stim_path = Path('../samediff_no-transf_tiny')
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # inspect_features(nets, dls)
