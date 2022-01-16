# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:02:00 2021

@author: mq20185996
"""

from imports import *


class add_fov_noise(Transform):
    def __init__(self, noise_mean, noise_sd, device):
        super(add_fov_noise, self).__init__()
        self.noise_mean = noise_mean
        self.noise_sd = noise_sd
        self.device = device

    def encodes(self, o):
        v = 100
        oo = []
        for i in range(len(o)):
            tmp = self.noise_mean + self.noise_sd * torch.randn(
                o[i][0][2].size(), device=self.device)
            tmp = TensorImage(tmp)
            oo.append((o[i][0].add((0, 0, tmp)), o[i][1]))
        return oo

    def decodes(self, o):
        return o


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
        return show_image(torch.cat([t1, line, t2, line, t3], dim=2),
                          ctx=ctx,
                          **kwargs)


def label_func(path):
    split_name = path.stem.split("_")
    return 0 if split_name[-1] == split_name[-2] else 1


def label_func_abstract(path):
    if path[0] == path[1]:
        label = 0
    else:
        label = 1

    return label


def label_func_class(path):
    split_name = path.stem.split("_")
    if "b" in split_name[-1]:
        label = 1
    elif "c" in split_name[-1]:
        label = 2
    elif "f" in split_name[-1]:
        label = 3
    elif "m" in split_name[-1]:
        label = 4
    else:
        print("error in class label")
        label = 0
    return label


def label_func_class_abstract(path):
    if "rect" in str(path[0]):
        label = 1
    elif "round" in str(path[0]):
        label = 2
    elif "spiky" in str(path[0]):
        label = 3
    else:
        print("error in class label")
        label = 0
    return label


def make_dls(
    stim_path,
    get_img_tuple_func,
    batch_sz=24,
    seed=0,
    test_prop=0.2,
    shuffle=True,
    lab_func=label_func,
):

    stim_path = Path(stim_path)
    pairs = glob.glob(os.path.join(stim_path, "*.png"))
    fnames = sorted(Path(s) for s in pairs)
    y = [lab_func(item) for item in fnames]

    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=seed,
                                 shuffle=True,
                                 stratify=y)

    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=lambda f: [[
            get_img_tuple_func(x, lab_func)[0],
            get_img_tuple_func(x, lab_func)[1],
            get_img_tuple_func(x, lab_func)[2],
            get_img_tuple_func(x, lab_func)[3],
        ] for x in f],
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
    )

    dls = siamese.dataloaders(
        fnames,
        bs=batch_sz,
        seed=seed,
        shuffle=shuffle,
        num_workers=0,
        device=defaults.device,
    )

    return dls


def make_dls_abstract(root,
                      get_img_tuple_func,
                      batch_sz=24,
                      seed=0,
                      test_prop=0.2,
                      shuffle=True,
                      lab_func=label_func_abstract):

    # func below calculate th distance (dissimilarity) between to stimuli [0:20]
    def calc_dist(pair):
        dist = 0
        file1 = os.path.basename(pair[0]).split("_")[1][:4]
        file2 = os.path.basename(pair[1]).split("_")[1][:4]
        for i in range(4):
            dist += abs(int(file1[i]) - int(file2[i]))
        return dist

    root = Path(root)
    cats = ["Cubies", "Smoothies", "Spikies"]

    dirs = sorted([os.path.join(root, cat) for cat in cats])

    stims = {
        os.path.split(dir_)[1]: sorted(glob.glob(os.path.join(dir_, "*.jpg")))
        for dir_ in dirs
    }

    # n stimuli in each category
    n_stims = len(stims[cats[1]])

    # list of all the 'same trials' (same stimulus, same category)
    same_trials = [[stims[cat][i], stims[cat][i]] for cat in cats
                   for i in range(n_stims)]

    # list of 'diff' trials (diff stimulus, same category)
    # here we build all the possible combinations indexes
    # we calculate all r-length tuples, in sorted order, no repeated elements e.g., AB AC AD BC BD CD
    diff_idxs = list(combinations(list(range(n_stims)), r=2))

    # here we calculate the differences between each possible combination of stimuli
    # only of one category, because the other categories will be the same
    dists = [
        calc_dist([stims[cats[0]][i], stims[cats[0]][j]]) for i, j in diff_idxs
    ]

    # create list of all the possible pairs of stimuli
    cubies = [(stims["Cubies"][i], stims["Cubies"][j]) for i, j in diff_idxs]
    smoothies = [(stims["Smoothies"][i], stims["Smoothies"][j])
                 for i, j in diff_idxs]
    spikies = [(stims["Spikies"][i], stims["Spikies"][j])
               for i, j in diff_idxs]

    # we zip pairs and distances, so we can select only first n stimuli (highest dist)
    zipped = list(sorted(zip(dists, cubies, smoothies, spikies), reverse=True))
    sliced = zipped[:n_stims]

    # here we make the stimuli list (same AND different)
    fnames = [(Path(pair[0]), Path(pair[1])) for item in sliced
              for pair in item if type(pair) is not int] + same_trials

    y = [fname1 == fname2 for [fname1, fname2] in fnames]

    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=42,
                                 shuffle=True,
                                 stratify=y)

    splits = splitter(fnames)

    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=lambda f: [[
            get_img_tuple_func(x, lab_func)[0],
            get_img_tuple_func(x, lab_func)[1],
            get_img_tuple_func(x, lab_func)[2],
            get_img_tuple_func(x, lab_func)[3],
        ] for x in f],
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
    )

    dls = siamese.dataloaders(
        fnames,
        bs=batch_sz,
        seed=seed,
        shuffle=True,
        device=defaults.device,
    )

    return dls


# TODO: implement this
def make_dls_imagenet():

    transform_item = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    path = Path(r'D:\Andrea_NN\data\IMAGENET\Compressed')

    dls = ImageDataLoaders.from_folder(
        path=path,
        valid='validation',  # val folder name
        train='train',  # train folder name
        device='cuda',
        item_tfms=transform_item,  # transforms to apply to an item
        batch_tfms=Normalize.from_stats(
            *imagenet_stats),  # transform to apply to the batch
        seed=42,
        bs=256,  # batch size
        shuffle=True  # shuffle training DataLoader
    )

    # See https://docs.fast.ai/vision.data.html#ImageDataLoaders or https://docs.fast.ai/tutorial.imagenette.html for examples


def get_tuples(files):
    return [[
        get_img_tuple_func(f)[0],
        get_img_tuple_func(f)[1],
        get_img_tuple_func(f)[2],
        get_img_tuple_func(f)[3],
    ] for f in files]


def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuple.create,
                          batch_tfms=IntToFloatTensor)


def get_x(t):
    return t[:3]


def get_y(t):
    return t[3]


def get_img_tuple_fov_empty_abstract(path, label_func):

    label = label_func(path)

    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    im1 = img1.convert("RGB").resize((224, 224))
    im2 = img2.convert("RGB").resize((224, 224))
    im3 = Image.new("RGB", (224, 224), (125, 125, 125))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def get_img_tuple_fov_diff_abstract(path, label_func):
    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    img3_basename = os.path.basename(path[0])
    cat = img3_basename.split("_")[0]
    cats = ["rect", "round", "spiky"]
    cats.remove(cat)
    new_id = f"{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}"
    img3_basename_new = (cats[random.randint(0, 1)] + "_" + new_id +
                         img3_basename.split("_")[1][4:])
    img3_path_split = os.path.join(
        os.path.split(path[0])[0], img3_basename_new).split(os.sep)
    img3_path_split[-2] = "*"
    img3_path_split[0] = img3_path_split[0] + os.sep
    img3_path = glob.glob(os.path.join(*img3_path_split))[0]
    img3 = Image.open(img3_path)

    if path[0] == path[1]:
        label = 0
    else:
        label = 1
    im1 = img1.convert("RGB").resize((224, 224))
    im2 = img2.convert("RGB").resize((224, 224))
    im3 = img3.convert("RGB").resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def get_img_tuple_fov_same_abstract(path, label_func):
    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    img3_basename = os.path.basename(path[0])
    new_id = f"{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}"
    img3_basename_new = (img3_basename.split("_")[0] + "_" + new_id +
                         img3_basename.split("_")[1][4:])
    img3 = Image.open(
        os.path.join(os.path.split(path[0])[0], img3_basename_new))

    if path[0] == path[1]:
        label = 0
    else:
        label = 1
    im1 = img1.convert("RGB").resize((224, 224))
    im2 = img2.convert("RGB").resize((224, 224))
    im3 = img3.convert("RGB").resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def get_img_tuple_fov_empty(path, label_func):
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


def get_img_tuple_fov_same(path, label_func):
    root = os.path.dirname(path)

    pair = Image.open(path)
    pair_basename = os.path.basename(path).split("_")

    fov_cat = pair_basename[-2][0]  # b
    fov_img_id = f"{fov_cat}{str(random.randint(1,30))}"  # b1
    fov_basename = pair_basename[:-2]
    fov_basename.extend([fov_img_id, fov_img_id + ".png"])
    fov_basename = "_".join(fov_basename)
    fov_path = os.path.join(root, fov_basename)
    fov_pair = Image.open(fov_path)

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
    im3 = fov_pair.crop((left2, top2, right2, bottom2)).resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def get_img_tuple_fov_diff(path, label_func):
    root = os.path.dirname(path)

    pair = Image.open(path)
    pair_basename = os.path.basename(path).split("_")
    per_cat = pair_basename[-2][0]  # c

    cats = ["b", "c", "f", "m"]
    cats.remove(per_cat)

    fov_cat = cats[random.randint(0, 2)]  # b
    fov_img_id = f"{fov_cat}{str(random.randint(1,30))}"  # b1
    fov_basename = pair_basename[:-2]
    fov_basename.extend([fov_img_id, fov_img_id + ".png"])
    fov_basename = "_".join(fov_basename)
    fov_path = os.path.join(root, fov_basename)
    fov_pair = Image.open(fov_path)

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
    im3 = fov_pair.crop((left2, top2, right2, bottom2)).resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )


def plot_cf(cf_y, cf_pred, cycle, epoch, path=""):
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


def train_networks(nets, criterion, dls, batch_sz, cycles, epochs, lr_min,
                   weight_decay, seed, condition):
    for net in nets:
        print(net.module.model_name)
        net.module.init_weights()
        net.module.init_pretrained_weights()
        net.module.freeze_pretrained_weights()
        params_to_update = net.parameters()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, params_to_update),
            lr=lr_min,
            weight_decay=weight_decay,
        )
        res = net.module.train_net(optimizer, criterion, dls, cycles, epochs)

        (tr_loss, tr_acc, te_loss, te_acc, cf_pred, cf_y) = res
        d = pd.DataFrame({
            "net": net.module.model_name,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "te_loss": te_loss,
            "te_acc": te_acc,
        })

        torch.save(net.state_dict(),
                   "net_111" + net.module.model_name + ".pth")
        d.to_csv("results_train_" + net.module.model_name + "_" + condition +
                 ".csv")


def test_noise(nets, criterion, stim_path, batch_sz, seed, condition):
    d = []
    noise_sd = np.linspace(0.0, 60.0, 25)
    for v in noise_sd:
        if condition == 'real_stim':
            dls = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz, seed)
        elif condition == 'abstract_stim':
            dls = make_dls_abstract(stim_path,
                                    get_img_tuple_fov_empty_abstract, batch_sz,
                                    seed)

        dls.to("cpu")
        dls.add_tfms([add_fov_noise(0, v, "cpu")], "before_batch", "valid")
        dls.to(defaults.device)
        # dls.valid.show_batch()
        # plt.show()
        # plt.close('all')

        for net in nets:
            print(net.module.model_name)
            net.load_state_dict(
                torch.load(
                    "net_111" + net.module.model_name + ".pth",
                    map_location=defaults.device,
                ))
            res = net.module.test_net(criterion, dls[1])
            (te_loss, te_acc, cf_pred, cf_y) = res
            d.append(
                pd.DataFrame({
                    "noise_sd": v,
                    "net": net.module.model_name,
                    "te_acc": te_acc
                }))
    d = pd.concat(d)
    d.to_csv("results_test_noise_" + condition + ".csv")

    return d


def test_fov_img(nets, criterion, stim_path, batch_sz, seed, condition):
    d_empty = []
    d_same = []
    d_diff = []

    if condition == 'real_stim':
        dls_empty = make_dls(stim_path, get_img_tuple_fov_empty, batch_sz,
                             seed)
        dls_same = make_dls(stim_path, get_img_tuple_fov_same, batch_sz, seed)
        dls_diff = make_dls(stim_path, get_img_tuple_fov_diff, batch_sz, seed)

    elif condition == 'abstract_stim':
        dls_empty = make_dls_abstract(stim_path,
                                      get_img_tuple_fov_empty_abstract,
                                      batch_sz, seed)
        dls_same = make_dls_abstract(stim_path,
                                     get_img_tuple_fov_same_abstract, batch_sz,
                                     seed)
        dls_diff = make_dls_abstract(stim_path,
                                     get_img_tuple_fov_diff_abstract, batch_sz,
                                     seed)

    for net in nets:
        print(net.module.model_name)
        state_dict = torch.load("net_111" + net.module.model_name + ".pth",
                                map_location=defaults.device)
        net.load_state_dict(state_dict)

        res_empty = net.module.test_net(criterion, dls_empty[1])
        res_same = net.module.test_net(criterion, dls_same[1])
        res_diff = net.module.test_net(criterion, dls_diff[1])

        d_empty.append(
            pd.DataFrame({
                "condition": "empty",
                "net": net.module.model_name,
                "te_acc": res_empty[1],
            }))
        d_same.append(
            pd.DataFrame({
                "condition": "same",
                "net": net.module.model_name,
                "te_acc": res_same[1],
            }))
        d_diff.append(
            pd.DataFrame({
                "condition": "diff",
                "net": net.module.model_name,
                "te_acc": res_diff[1],
            }))

    d_empty = pd.concat(d_empty)
    d_same = pd.concat(d_same)
    d_diff = pd.concat(d_diff)
    d = [d_empty, d_same, d_diff]
    d = pd.concat(d)
    d.to_csv("results_test_fovimg_" + condition + ".csv")


def get_features(net, net_layer, net_layer_name, dls):

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    handle = net_layer[0].register_forward_hook(get_activation(net_layer_name))

    X = []
    y = []

    net.to(defaults.device)
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in dls[1]:
            print(labels)
            out = net(inputs)
            X.append(activation[net_layer_name].to("cpu").numpy())
            y.append(labels.to("cpu").numpy())
    return X, y


def test_classify(nets, criterion, stim_path, batch_sz, seed, condition):

    if condition == 'real_stim':
        dls = make_dls(stim_path,
                       get_img_tuple_fov_empty,
                       batch_sz,
                       seed,
                       0.2,
                       lab_func=label_func_class)

    elif condition == 'abstract_stim':
        dls = make_dls_abstract(stim_path,
                                get_img_tuple_fov_empty_abstract,
                                batch_sz,
                                seed,
                                0.2,
                                lab_func=label_func_class_abstract)

    res = []
    for net in nets:
        print(net.module.model_name)
        state_dict = torch.load('net_111' + net.module.model_name + '.pth',
                                map_location=defaults.device)
        net.load_state_dict(state_dict)
        net = net.module.to('cpu')

        net_layer = net.V1_fov
        net_layer_name = 'fov'
        X, y = get_features(net, net_layer, net_layer_name, dls)
        # for i in range(3):
        #     plt.imshow(X[i][0, 1, :, :])
        #     plt.show()
        X = np.vstack(X)
        X = X.reshape(X.shape[0], -1)
        y = np.hstack(y)

        if condition == 'real_stim':
            cb_mask = np.isin(y, [0, 1])
            mf_mask = np.isin(y, [2, 3])

            print(np.unique(y))

            y_all = y
            y_cb = y[cb_mask]
            y_mf = y[mf_mask]
            y_fv = np.array(y)
            y_fv[cb_mask] = 5
            y_fv[mf_mask] = 6

            print(np.unique(y))
            print(np.unique(y_cb))
            print(np.unique(y_mf))
            print(np.unique(y_fv))
            print(np.unique(y_all))

            X_all = X
            X_cb = X[cb_mask, :]
            X_mf = X[mf_mask, :]
            X_fv = X

            Xy_dict = {
                'cb': [X_cb, y_cb],
                'mf': [X_mf, y_mf],
                'fv': [X_fv, y_fv],
                'all': [X_all, y_all]
            }

        elif condition == 'abstract_stim':
            Xy_dict = {
                'all': [X, y],
            }

        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        skf = StratifiedKFold(n_splits=5)

        for key, Xy in Xy_dict.items():
            X = Xy[0]
            y = Xy[1]

            f = 0
            for train_index, test_index in skf.split(X, y):
                print(f)
                f += 1
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                pipe.fit(X_train, y_train)
                acc = pipe.score(X_test, y_test)

                res.append(
                    pd.DataFrame(
                        {
                            'key': key,
                            'net': net.model_name,
                            'fold': f,
                            'acc': acc
                        },
                        index=[f]))

    res = pd.concat(res)
    res.to_csv('results_test_classify_' + condition + '.csv')


def inspect_test_fov_img():
    d_real = pd.read_csv('results_test_fovimg_real_stim.csv')
    d_abstract = pd.read_csv('results_test_fovimg_abstract_stim.csv')

    fig, ax = plt.subplots(1, 2, squeeze=False)
    sn.barplot(data=d_real,
               x="condition",
               y="te_acc",
               hue="condition",
               ax=ax[0, 0])
    sn.barplot(data=d_abstract,
               x="condition",
               y="te_acc",
               hue="condition",
               ax=ax[0, 1])
    plt.tight_layout()
    plt.savefig("results_test_fovimg_real_and_abstract.pdf")
    plt.close()


def inspect_test_noise():
    d_real = pd.read_csv('results_test_noise_real_stim.csv')
    d_abstract = pd.read_csv('results_test_noise_abstract_stim.csv')

    d_real['condition'] = 'real_stim'
    d_abstract['condition'] = 'abstract_stim'

    d = pd.concat((d_real, d_abstract))

    sn.scatterplot(data=d, x="noise_sd", y="te_acc", hue="condition")
    plt.savefig('results_test_noise_real_and_abstract.pdf')
    plt.close()


def inspect_test_classify():
    d_all = pd.read_csv('results_test_classify_all.csv')
    d_cb = pd.read_csv('results_test_classify_cb.csv')
    d_fv = pd.read_csv('results_test_classify_fv.csv')
    d_mf = pd.read_csv('results_test_classify_mf.csv')

    d_all['class'] = 'all'
    d_cb['class'] = 'cb'
    d_fv['class'] = 'fv'
    d_mf['class'] = 'mf'

    d_all['condition'] = 'real_stim'
    d_cb['condition'] = 'real_stim'
    d_fv['condition'] = 'real_stim'
    d_mf['condition'] = 'real_stim'

    d_abstract = pd.read_csv('results_test_classify_all_abstract_stim.csv')
    d_abstract['class'] = 'abstract'
    d_abstract['condition'] = 'abstract_stim'

    d = pd.concat((d_all, d_cb, d_fv, d_mf))

    sn.barplot(data=d, x='net', y='acc', hue='class')
    plt.savefig('results_test_classify_real_and_abstract.pdf')
    plt.close()


def inspect_features(nets, dls):
    def plot_weights(w, title):
        fig, ax = plt.subplots(3, 20, squeeze=False, figsize=(10, 4))
        for j in range(20):
            ax[0, j].imshow(w[j, 0, :, :])
            ax[1, j].imshow(w[j, 1, :, :])
            ax[2, j].imshow(w[j, 2, :, :])
        [a.set_xticks([]) for a in ax.flatten()]
        [a.set_yticks([]) for a in ax.flatten()]
        fig.suptitle(title)
        plt.show()

    # select
    w_name = "V1_fov.0.weight"
    x_key = "V1_fov"

    # random init
    net = SiameseNet13(w_dropout_1, w_dropout_2)
    net.init_weights()
    state_dict = net.state_dict()
    w = state_dict[w_name]
    net.to(defaults.device)
    net = nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in dls[0]:
            out = net(inputs)
            plot_weights(w, "Random initial weights")
    # corenet preload
    net = SiameseNet13(w_dropout_1, w_dropout_2)
    net.init_weights()
    net.init_pretrained_weights()
    state_dict = net.state_dict()
    w = state_dict[w_name]
    net.to(defaults.device)
    net = nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in dls[0]:
            out = net(inputs)
            plot_weights(w, "Corenet pretrained weights")
    # custom training
    net = SiameseNet13(w_dropout_1, w_dropout_2)
    net.init_weights()
    net.init_pretrained_weights()
    net = nn.DataParallel(net)
    net.module.init_trained_weights()
    net = net.module.to("cpu")
    state_dict = net.state_dict()
    w = state_dict[w_name]
    net.to(defaults.device)
    net = nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in dls[0]:
            out = net(inputs)
            plot_weights(w, "Custom trained weights")


def show_triplets(dls):
    print("Showing triplets from the first batch...")
    batch = dls.one_batch()
    for i in range(len(batch[0][0])):
        im1 = batch[0][0][i]
        im2 = batch[0][1][i]
        im3 = batch[0][2][i]

        imgs = torch.cat([im1, im2, im3], dim=2)

        y = batch[1][i].item()
        if y == 0:
            label = str(i) + " - Same"
        else:
            label = str(i) + " - Different"
        show_image(imgs, title=y)
