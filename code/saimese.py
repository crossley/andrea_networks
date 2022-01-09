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

    epochs = 25
    cycles = 1
    batch_sz = 24
    lr_min = 1e-5
    weight_decay = 1e-3
    w_dropout_1 = 0.8
    w_dropout_2 = 0.8
    test_prop = 0.2

    # net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    # net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    # net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    # net_02 = SiameseNet02(w_dropout_1, w_dropout_2)
    # net_12 = SiameseNet12(w_dropout_1, w_dropout_2)
    # net_22 = SiameseNet22(w_dropout_1, w_dropout_2)
    net_13 = SiameseNet13(w_dropout_1, w_dropout_2)

    # nets = [net_0, net_1, net_2, net_02, net_12, net_22, net_13]
    # nets = [x.to(defaults.device) for x in nets]
    # nets = [nn.DataParallel(x) for x in nets]

    net_13.to(defaults.device)
    net_13 = nn.DataParallel(net_13)
    
    nets = [net_13]

    # criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyLossFlat()

    # NOTE: fmri stim training / testing
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # train_networks([net_13], criterion, dls, batch_sz, cycles, epochs, lr_min,
    #                 weight_decay, seed)
    # test_fov_img(nets, criterion, stim_path, batch_sz, seed)
    # test_noise(nets, criterion, stim_path, batch_sz, seed)
    # test_classify(nets, criterion, stim_path, batch_sz, seed)

    # NOTE: abstract stim training / testing
    dls = make_dls_abstract(stim_path_abstract, get_img_tuple_fov_empty_abstract,
                            batch_sz, seed)
                            
    show_triplets(dls)
    
    for (inputs, labels) in dls[1]:
        print(labels)
        print(inputs[0].shape)
    # train_networks(nets, criterion, dls, batch_sz, cycles, epochs, lr_min,
    #                weight_decay, seed)

    # inspect weights / features
    # stim_path = Path('../samediff_no-transf_tiny')
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # inspect_features(nets, dls)

# TODO:
# - what is get_img_tuple_fov_diff_fv?
# - what is this all about: 'need to pass fnames1 and fnames2 to make images

# TODO:
# - train on abstract stimuli
# - sort out visualisations to inspect the effect of training
# - Try different feedback architectures
# - Deep net to fMRI mapping

# TODO: Pour over these papers and test our model on anything appropriate
# Weldon, K. B., Rich, A. N., Woolgar, A., & Williams, M. A. (2016). Disruption
# of foveal space impairs discrimination of peripheral objects. Frontiers in
# psychology, 7, 699.

# Fan, X., Wang, L., Shao, H., Kersten, D., & He, S. (2016). Temporally flexible
# feedback signal to foveal cortex for peripheral object recognition. Proceedings
# of the National Academy of Sciences, 113(41), 11627-11632. Chicago

# Yu, Q., & Shim, W. M. (2016). Modulating foveal representation can influence
# visual discrimination in the periphery. Journal of Vision, 16(3), 15-15.
# Chicago
