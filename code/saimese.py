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
    epochs = 2
    cycles = 1
    batch_sz = 24
    lr_min = 1e-4
    weight_decay = 1e-3
    w_dropout_1 = 0.8
    w_dropout_2 = 0.8

    data_loader = make_dls(stim_path, batch_sz, seed)

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    net_3 = SiameseNet12(w_dropout_1, w_dropout_2)
    net_4 = SiameseNet22(w_dropout_1, w_dropout_2)

    nets = [net_0, net_1, net_2, net_3, net_4]

    nets = [x.to(defaults.device) for x in nets]
    nets = [nn.DataParallel(x) for x in nets]

    criterion = nn.CrossEntropyLoss()

    # NOTE: training
    for net in nets:
        print(net.module.model_name)
        net.module.init_weights()
        net.module.init_pretrained_weights()
        net.module.freeze_pretrained_weights()
        params_to_update = net.parameters()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      params_to_update),
                               lr=lr_min,
                               weight_decay=weight_decay)
        net.module.train_net(optimizer, criterion, data_loader, cycles, epochs)
        torch.save(net.state_dict(),
                   'net_111' + net.module.model_name + '.pth')

    # NOTE: test sensitivity to fovea noise
    noise_levels = np.linspace(0.0, 1.0, 2)
    test_loader = data_loader[1]
    for net in nets:
        print(net.module.model_name)
        net.load_state_dict(
            torch.load('net_111' + net.module.model_name + '.pth'))

        for v in noise_levels:
            for (inputs, labels) in test_loader:
                X = []
                y = []
                fov_img = inputs[2]                
                X.append(fov_img + v * torch.randn(fov_img.shape, device='cuda'))
                y.append(labels)
    
                test_result = net.module.test_net(criterion, X, y)

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
