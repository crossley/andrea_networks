# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:02:00 2021

@author: mq20185996
"""

from imports import *


def train_nets():

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

    dls = make_dls(stim_path, batch_sz, seed)
    train_loader = dls.train
    test_loader = dls.valid

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    net_3 = SiameseNet12(w_dropout_1, w_dropout_2)
    net_4 = SiameseNet22(w_dropout_1, w_dropout_2)

    nets = [net_0, net_1, net_2, net_3, net_4]

    [init_weights(x) for x in nets]
    [init_pretrained_weights(x) for x in nets]

    params_to_update = [x.parameters() for x in nets]

    optimizers = [
        optim.Adam(filter(lambda p: p.requires_grad, x),
                   lr=lr_min,
                   weight_decay=weight_decay) for x in params_to_update
    ]

    nets = [nn.DataParallel(x) for x in nets]

    [x.to(defaults.device) for x in nets]

    criterion = nn.CrossEntropyLoss()

    p = (criterion, cycles, epochs, train_loader, test_loader)

    res = [train_net(nets[x], optimizers[x], p) for x in range(len(nets))]

    [
        torch.save(x.state_dict(), 'net_' + x.module.model_name + '.pth')
        for x in nets
    ]


def train_net(net, optimizer, p):

    criterion = p[0]
    cycles = p[1]
    epochs = p[2]
    train_loader = p[3]
    test_loader = p[4]

    for cycle in range(cycles):
        tr_loss = []
        tr_acc = []
        te_loss = []
        te_acc = []

        for epoch in range(epochs):
            # TRAIN
            net.train()

            tr_running_loss = 0.0
            tr_correct = 0
            tr_total = 0
            start = time.time()
            for (inputs, labels) in train_loader:
                optimizer.zero_grad()
                out = net(inputs)
                _, pred = torch.max(out, 1)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                tr_running_loss += loss.item()
                tr_total += labels.size(0)
                tr_correct += (pred == labels).sum().item()

            tr_loss.append(tr_running_loss)
            tr_acc.append(100 * tr_correct / tr_total)

            # TEST
            net.eval()

            te_running_loss = 0.0
            te_correct = 0
            te_total = 0
            cf_pred = []
            cf_y = []
            with torch.no_grad():
                for (inputs, labels) in test_loader:
                    out = net(inputs)
                    _, pred = torch.max(out, 1)
                    loss = criterion(out, labels)
                    te_running_loss += loss.item()
                    te_total += labels.size(0)
                    te_correct += (pred == labels).sum().item()
                    cf_y += labels.cpu().detach().tolist()
                    cf_pred += pred.cpu().detach().tolist()

                te_acc.append(100 * te_correct / te_total)
                te_loss.append(te_running_loss)
                end = time.time() - start
                print("{0:0.2f}".format(cycle + 1),
                      "{0:0.2f}".format(epoch + 1),
                      "{0:0.2f}".format(tr_running_loss),
                      "{0:0.2f}".format(te_running_loss),
                      "{0:0.2f}".format(100 * tr_correct / tr_total),
                      "{0:0.2f}".format(100 * te_correct / te_total),
                      "{0:0.2f}".format(end))

    return (tr_loss, tr_acc, te_loss, te_acc, cf_pred, cf_y)
