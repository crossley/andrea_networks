# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:02:00 2021

@author: mq20185996
"""

from imports import *


def label_func(path):
    split_name = path.stem.split("_")
    return 0 if split_name[-1] == split_name[-2] else 1


def get_img_tuple_no_noise(path):
    pair = Image.open(path)

    label = label_func(Path(path))
    orientation = os.path.basename(path).split("_")[-3]

    width, height = pair.size

    if orientation == "normal":
        left1, top1, right1, bottom1 = width - width // 4, 0, width, height // 4
        left2, top2, right2, bottom2 = 0, height - height // 4, width // 4, height
    else:
        left1, top1, right1, bottom1 = 0, 0, width // 4, height // 4
        left2, top2, right2, bottom2 = (
            width - width // 4,
            height - height // 4,
            width,
            height,
        )

    im1 = pair.crop((left1, top1, right1, bottom1)).resize((224, 224))
    im2 = pair.crop((left2, top2, right2, bottom2)).resize((224, 224))
    im3 = Image.new("RGB", (224, 224), (125, 125, 125))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def get_img_tuple_noise(path):

    pair = Image.open(path)

    label = label_func(Path(path))
    orientation = os.path.basename(path).split("_")[-3]

    width, height = pair.size

    if orientation == "normal":
        left1, top1, right1, bottom1 = width - width // 4, 0, width, height // 4
        left2, top2, right2, bottom2 = 0, height - height // 4, width // 4, height
    else:
        left1, top1, right1, bottom1 = 0, 0, width // 4, height // 4
        left2, top2, right2, bottom2 = (
            width - width // 4,
            height - height // 4,
            width,
            height,
        )

    im1 = pair.crop((left1, top1, right1, bottom1)).resize((224, 224))
    im2 = pair.crop((left2, top2, right2, bottom2)).resize((224, 224))
    im3 = Image.new("RGB", (224, 224), (125, 125, 125))
    im3 = Image.fromarray(
        np.uint8(
            skimage.util.random_noise(
                skimage.img_as_float(im3), mode="s&p", amount=1) * 255))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


class ImageTuple(fastuple):
    @classmethod
    def create(cls, fns):
        return cls(fns)

    def show(self, ctx=None, **kwargs):
        t1, t2, t3 = self
        if (not isinstance(t1, Tensor) or not isinstance(t2, Tensor)
                or t1.shape != t2.shape):
            return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1, line, t2], dim=2), ctx=ctx, **kwargs)


def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuple.create,
                          batch_tfms=IntToFloatTensor)


def get_tuples_no_noise(files):

    return [[
        get_img_tuple_no_noise(f)[0],
        get_img_tuple_no_noise(f)[1],
        get_img_tuple_no_noise(f)[2],
        get_img_tuple_no_noise(f)[3],
    ] for f in files]


def get_tuples_noise(files):
    return [[
        get_img_tuple_noise(f)[0],
        get_img_tuple_noise(f)[1],
        get_img_tuple_noise(f)[2],
        get_img_tuple_noise(f)[3],
    ] for f in files]


def get_x(t):
    return t[:3]


def get_y(t):
    return t[3]


def make_dls(stim_path, batch_sz=24, seed=0, test_prop=0.2):

    stim_path = Path(stim_path)
    pairs = glob.glob(os.path.join(stim_path, "*.png"))
    fnames = sorted(Path(s) for s in pairs)
    y = [label_func(item) for item in fnames]

    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=42,
                                 shuffle=True,
                                 stratify=y)
    splits = splitter(fnames)
    # splits = RandomSplitter()(fnames)
    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=get_tuples_no_noise,
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
    )

    dls = siamese.dataloaders(
        fnames,
        bs=batch_sz,
        seed=seed,
        # shuffle=True,
        # device = 'cuda',
    )

    # check that train and test splits have balanced classes
    train_test = ["TRAIN", "TEST"]
    for train_test_id in [0, 1]:
        s = 0
        d = 0
        for item in dls.__getitem__(train_test_id).items:
            # print(label_from_path(item))
            # print(item)
            # print('---')
            if item[3] == 1:
                s += 1
            else:
                d += 1
        print(
            f"{train_test[train_test_id]} SET (same, diff): {str(s)}, {str(d)}"
        )
    return dls


def plot_filters_multi_channel(t, path=""):

    # get the number of kernels
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)

        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    if path != "":
        plt.savefig(path)
    plt.show()


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return net


def make_cf(cf_y, cf_pred, cycle, epoch, path=""):
    plt.figure(figsize=(7, 7))
    cf_matrix = confusion_matrix(cf_y, cf_pred)
    df_cm = pd.DataFrame(
        cf_matrix,
        index=[i for i in ["Same", "Different"]],
        columns=[i for i in ["Same", "Different"]],
    )
    sn.heatmap(df_cm, annot=True, cbar=False, cmap="Blues", fmt="d")
    plt.suptitle(f"Epoch {cycle+1} x {epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if path != "":
        plt.savefig(os.path.join(path, f"cm_{cycle+1}x{epoch+1}.png"))
    plt.show()


def plot_losses(tr_loss, te_loss, cycle, epoch, path=""):
    plt.plot(tr_loss, label="Train")
    plt.plot(te_loss, label="Test")
    plt.suptitle(f"Losses\nEpoch {cycle+1} x {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if path != "":
        plt.savefig(os.path.join(path, f"loss_{cycle+1}x{epoch+1}.png"))
    plt.show()


def plot_acc(tr_acc, te_acc, cycle, epoch, path=""):
    plt.plot(tr_acc, label="Train")
    plt.plot(te_acc, label="Test")
    plt.suptitle(f"Accuracy\nEpoch {cycle+1} x {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if path != "":
        plt.savefig(os.path.join(path, f"acc_{cycle+1}x{epoch+1}.png"))
    plt.show()


def init_pretrained_weights(net):
    # use corenet weights pretrained on imagenet
    url = f'https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth'
    ckpt_data = torch.utils.model_zoo.load_url(
        url, map_location="cuda" if torch.cuda.is_available() else "cpu")

    state_dict = {
        "V1.0.weight": ckpt_data['state_dict']['module.V1.conv.weight'],
        "V1.0.bias": ckpt_data['state_dict']['module.V1.conv.bias'],
        "V2.0.weight": ckpt_data['state_dict']['module.V2.conv.weight'],
        "V2.0.bias": ckpt_data['state_dict']['module.V2.conv.bias'],
        "V4.0.weight": ckpt_data['state_dict']['module.V4.conv.weight'],
        "V4.0.bias": ckpt_data['state_dict']['module.V4.conv.bias'],
        "IT.0.weight": ckpt_data['state_dict']['module.IT.conv.weight'],
        "IT.0.bias": ckpt_data['state_dict']['module.IT.conv.bias'],
    }

    net.load_state_dict(state_dict, strict=False)

    # hold all saimese trunk weights to their pretrained values
    net.V1[0].weight.requires_grad = False
    net.V1[0].bias.requires_grad = False
    net.V2[0].weight.requires_grad = False
    net.V2[0].bias.requires_grad = False
    net.V4[0].weight.requires_grad = False
    net.V4[0].bias.requires_grad = False
    net.IT[0].weight.requires_grad = False
    net.IT[0].bias.requires_grad = False

    # learn periphery to fovea feedback weights
    net.fb[0].weight.requires_grad = True
    net.fb[0].bias.requires_grad = True

    return net


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
                print(
                    cycle + 1,
                    epoch + 1,
                    tr_running_loss,
                    te_running_loss,
                    100 * tr_correct / tr_total,
                    100 * te_correct / te_total,
                )

    return (tr_loss, tr_acc, te_loss, te_acc, cf_pred, cf_y)


def test_net(net, p):

    cycles = p[0]
    epochs = p[1]
    train_loader = p[2]
    test_loader = p[3]
    noise_vars = p[4]

    res = []

    for v in noise_vars:

        for cycle in range(cycles):

            tr_loss = []
            tr_acc = []
            te_loss = []
            te_acc = []

            for epoch in range(epochs):

                net.eval()

                te_running_loss = 0.0
                te_correct = 0
                te_total = 0
                cf_pred = []
                cf_y = []
                with torch.no_grad():
                    start = time.time()
                    for (inputs, labels) in train_loader:

                        inputs = (
                            inputs[0], inputs[1], inputs[2] +
                            v * torch.randn(inputs[2].shape, device='cuda'))

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
                    print(cycle + 1, epoch + 1, te_running_loss,
                          100 * te_correct / te_total)

        d = pd.DataFrame({
            'noise_var': v,
            'te_loss': te_loss,
            'te_acc': te_acc
        })
        res.append(d)

    res = pd.concat(res)

    return res


def inspect_results_test(res):

    fig, ax = plt.subplots(1, len(res), figsize=(12, 4), squeeze=False)

    for i in range(len(res)):
        sn.violinplot(data=res[i], x='noise_var', y='te_acc', ax=ax[0, i])

    plt.tight_layout()
    plt.show()


def inspect_results(res):

    tr_loss = res[0]
    tr_acc = res[1]
    te_loss = res[2]
    te_acc = res[3]
    cf_pred = res[4]
    cf_y = res[5]

    fig, ax = plt.subplots(1, 3, figsize=(10, 4), squeeze=False)

    cf_matrix = confusion_matrix(cf_y, cf_pred)
    df_cm = pd.DataFrame(
        cf_matrix,
        index=[i for i in ["Same", "Different"]],
        columns=[i for i in ["Same", "Different"]],
    )
    sn.heatmap(df_cm,
               annot=True,
               cbar=False,
               cmap="Blues",
               fmt="d",
               ax=ax[0, 0])
    ax[0, 0].set_xlabel("Predicted")
    ax[0, 0].set_ylabel("Actual")

    ax[0, 1].plot(tr_loss, label="Train")
    ax[0, 1].plot(te_loss, label="Test")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("Loss")

    ax[0, 2].plot(tr_acc, label="Train")
    ax[0, 2].plot(te_acc, label="Test")
    ax[0, 2].set_xlabel("Epoch")
    ax[0, 2].set_ylabel("Accuracy")

    plt.legend()
    plt.tight_layout()
    plt.show()


def train_nets(p):

    stim_path = p[0]
    epochs = p[1]
    cycles = p[2]
    batch_sz = p[3]
    lr_min = p[4]
    weight_decay = p[5]
    w_dropout_1 = p[6]
    w_dropout_2 = p[7]
    seed = p[8]

    dls = make_dls(stim_path, batch_sz, seed)
    train_loader = dls.train
    test_loader = dls.valid
    # dls.show_batch(max_n = 2)
    # plt.show()

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

    [torch.save(x.state_dict(), 'net_' + x.model_name + '.pth') for x in nets]

    inspect_results(res)


def test_nets_noise():

    net_0 = SiameseNet0(w_dropout_1, w_dropout_2)
    net_1 = SiameseNet1(w_dropout_1, w_dropout_2)
    net_2 = SiameseNet2(w_dropout_1, w_dropout_2)
    net_3 = SiameseNet12(w_dropout_1, w_dropout_2)
    net_4 = SiameseNet22(w_dropout_1, w_dropout_2)

    nets = [net_0, net_1, net_2, net_3, net_4]
    nets = [nn.DataParallel(x) for x in nets]

    [
        x.load_state_dict(torch.load('net_' + x.module.model_name + '.pth'))
        for x in nets
    ]

    [x.to('cuda') for x in nets]

    cycles = 1
    epochs = 1
    noise_levels = np.linspace(0.0, 3.0, 30)
    batch_sz = test_loader.n

    p = (cycles, epochs, train_loader, test_loader, noise_levels)

    res = [test_net(x, p) for x in nets]

    inspect_results_test(res)
