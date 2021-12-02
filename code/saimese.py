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
    # stim_path = Path(r'D:\Andrea_NN\stimuli\samediff')
    stim_path = Path('../samediff_no-transf')
    epochs = 2
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

    # NOTE: training
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # for net in nets:
    #     print(net.module.model_name)
    #     net.module.init_weights()
    #     net.module.init_pretrained_weights()
    #     net.module.freeze_pretrained_weights()
    #     params_to_update = net.parameters()
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad,
    #                                   params_to_update),
    #                             lr=lr_min,
    #                             weight_decay=weight_decay)
    #     net.module.train_net(optimizer, criterion, dls, cycles, epochs)
    #     torch.save(net.state_dict(),
    #                 'net_111' + net.module.model_name + '.pth')

    # NOTE: test sensitivity to fovea noise
    # noise_sd = np.linspace(0.0, 100.0, 2)
    # print(noise_sd)
    # for v in noise_sd:
    #     dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    #     dls.add_tfms([add_fov_noise(0, v)], 'before_batch', 'valid')
    #     # dls.valid.show_batch()
    #     # plt.show()
    #     # plt.close('all')

    #     for net in nets:
    #         print(net.module.model_name)
    #         net.load_state_dict(
    #             torch.load('net_111' + net.module.model_name + '.pth',
    #                        map_location=torch.device('cpu')))

    #         res = net.module.test_net(criterion, dls[1])

    # NOTE: test sensitivity to fovea images (same vs diff)
    # dls_same = make_dls(stim_path, get_img_tuple_fov_same, batch_sz, seed)
    # dls_diff = make_dls(stim_path, get_img_tuple_fov_diff, batch_sz, seed)
    # for net in nets:
    #     print(net.module.model_name)
    #     net.load_state_dict(
    #         torch.load('net_111' + net.module.model_name + '.pth',
    #                    map_location=torch.device('cpu')))

    #     res_same = net.module.test_net(criterion, dls_same[1])
    #     res_diff = net.module.test_net(criterion, dls_diff[1])

    # NOTE: test ability to decode from periphery and fovea areas
    # dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
    # for net in nets:
    #     net = nets[1]
    #     print(net.module.model_name)
    #     # net.load_state_dict(
    #     #     torch.load('net_111' + net.module.model_name + '.pth',
    #     #                map_location=torch.device(defaults.device)))
    #     net.load_state_dict(
    #         torch.load('net_111' + net.module.model_name + '.pth',
    #                    map_location='cpu'))
    #     net = net.module.to('cpu')

    #     activation = {}

    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activation[name] = output.detach()

    #         return hook

    #     handle = net.fb[0].register_forward_hook(get_activation('fb'))

    #     X = []
    #     y = []

    #     net.eval()
    #     with torch.no_grad():
    #         for (inputs, labels) in dls[0]:
    #             out = net(inputs)
    #             X.append(activation['fb'].numpy())
    #             y.append(labels.numpy())

    #     X = np.vstack(X)
    #     X = X.reshape(X.shape[0], -1)
    #     y = np.hstack(y)

    #     # pipe = make_pipeline(StandardScaler(), SVC())
    #     pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

    #     skf = StratifiedKFold(n_splits=5)

    #     res = []
    #     f = 0
    #     for train_index, test_index in skf.split(X, y):
    #         f += 1
    #         X_train, X_test = X[train_index, :], X[test_index, :]
    #         y_train, y_test = y[train_index], y[test_index]
    #         pipe.fit(X_train, y_train)
    #         acc = pipe.score(X_test, y_test)
    #         d = pd.DataFrame({'fold': f, 'acc': acc}, index=[f])
    #         res.append(d)

    #     res = pd.concat(res)

# get_img_tuple_fov_empty
# get_img_tuple_fov_same
# get_img_tuple_fov_diff
# get_img_tuple_abstract
# get_img_tuple_abstract_fov_diff
# get_img_tuple_abstract_fov_same

# TODO:
# Try different feedback architectures
# - fully interconnected vs conv
# Deep net to fMRI mapping
