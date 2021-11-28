# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:02:00 2021

@author: mq20185996
"""

from imports import *


def label_func(path):
    split_name = path.stem.split("_")
    return 0 if split_name[-1] == split_name[-2] else 1


def get_x(t):
    return t[:3]


def get_y(t):
    return t[3]


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

def make_dls_abstract(root, batch_sz=24, seed=0, test_prop=0.2):

    root = Path(root)
    cats = ['Cubies', 'Smoothies', 'Spikies']

    dirs = [os.path.join(root, cat) for cat in cats]
    stims = {os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
             for dir_ in dirs}
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    def calc_dist(pair):
        dist = 0
        file1 = os.path.basename(pair[0]).split('_')[1][:4]
        file2 = os.path.basename(pair[1]).split('_')[1][:4]
        for i in range(4):
            dist += abs(int(file1[i]) - int(file2[i]))
        return dist

    fnames = [[fnames1[i],fnames2[i]] for i in range(len(fnames1)) if calc_dist(pair) > 5]

    # TODO: need to pass fnames1 and fnames2 to make images
    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=42,
                                 shuffle=True,
                                 stratify=y)

    splits = splitter(fnames)

    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=get_tuples_abstract,
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

def make_dls_abstract_fov_diff(root, batch_sz=24, seed=0, test_prop=0.2):

    root = Path(root)
    cats = ['Cubies', 'Smoothies', 'Spikies']

    dirs = [os.path.join(root, cat) for cat in cats]
    stims = {os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
             for dir_ in dirs}
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    fnames = [[fnames1[i],fnames2[i]] for i in range(len(fnames1))]

    # TODO: need to pass fnames1 and fnames2 to make images
    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=42,
                                 shuffle=True,
                                 stratify=y)

    splits = splitter(fnames)

    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=get_tuples_abstract_fov_diff,
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

def make_dls_abstract_fov_same(root, batch_sz=24, seed=0, test_prop=0.2):

    root = Path(root)
    cats = ['Cubies', 'Smoothies', 'Spikies']

    dirs = [os.path.join(root, cat) for cat in cats]
    stims = {os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
             for dir_ in dirs}
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    fnames = [[fnames1[i],fnames2[i]] for i in range(len(fnames1))]

    # TODO: need to pass fnames1 and fnames2 to make images
    splitter = TrainTestSplitter(test_size=test_prop,
                                 random_state=42,
                                 shuffle=True,
                                 stratify=y)

    splits = splitter(fnames)

    siamese = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        get_items=get_tuples_abstract_fov_same,
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

def get_tuples_abstract(files):
    return [[
        get_img_tuple_abstract(f)[0],
        get_img_tuple_abstract(f)[1],
        get_img_tuple_abstract(f)[2],
        get_img_tuple_abstract(f)[3],
    ] for f in files]

def get_tuples_abstract_fov_diff(files):
    return [[
        get_img_tuple_abstract_fov_diff(f)[0],
        get_img_tuple_abstract_fov_diff(f)[1],
        get_img_tuple_abstract_fov_diff(f)[2],
        get_img_tuple_abstract_fov_diff(f)[3],
    ] for f in files]

def get_tuples_abstract_fov_same(files):
    return [[
        get_img_tuple_abstract_fov_same(f)[0],
        get_img_tuple_abstract_fov_same(f)[1],
        get_img_tuple_abstract_fov_same(f)[2],
        get_img_tuple_abstract_fov_same(f)[3],
    ] for f in files]

def get_img_tuple_abstract(path):
    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    if path[0] == path[1]:
        label = 0
    else:
        label = 1

    im1 = img1.resize((224, 224))
    im2 = img2.resize((224, 224))
    im3 = Image.new("RGB", (224, 224), (125, 125, 125))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )

def get_img_tuple_abstract_fov_diff(path):
    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    img3_basename = os.path.basename(path[0])
    cat = img3_basename.split('_')[0]
    cats = ['rect','round','spiky']
    cats.remove(cat)
    new_id = f'{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}'
    img3_basename_new = cats[random.randint(0,1)] + '_' + new_id + img3_basename.split('_')[1][4:]
    img3_path_split = os.path.join(os.path.split(path[0])[0], img3_basename_new).split(os.sep)
    img3_path_split[-2] = '*'
    img3_path_split[0] = img3_path_split[0] + os.sep
    img3_path = glob.glob(os.path.join(*img3_path_split))[0]
    img3 = Image.open(img3_path)

    if path[0] == path[1]:
        label = 0
    else:
        label = 1

    im1 = img1.resize((224, 224))
    im2 = img2.resize((224, 224))
    im3 = img3.resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )

def get_img_tuple_abstract_fov_same(path):
    img1 = Image.open(path[0])
    img2 = Image.open(path[1])

    img3_basename = os.path.basename(path[0])
    new_id = f'{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}{random.randint(0,5)}'
    img3_basename_new = img3_basename.split('_')[0] + '_' + new_id + img3_basename.split('_')[1][4:]
    img3 = Image.open(os.path.join(os.path.split(path[0])[0], img3_basename_new))

    if path[0] == path[1]:
        label = 0
    else:
        label = 1

    im1 = img1.resize((224, 224))
    im2 = img2.resize((224, 224))
    im3 = img3.resize((224, 224))

    return (
        ToTensor()(PILImage(im1)),
        ToTensor()(PILImage(im2)),
        ToTensor()(PILImage(im3)),
        label,
    )

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


def get_tuples_fov_same(files):
    return [[
        get_img_tuple_fov_same(f)[0],
        get_img_tuple_fov_same(f)[1],
        get_img_tuple_fov_same(f)[2],
        get_img_tuple_fov_same(f)[3],
    ] for f in files]


def get_tuples_fov_diff(files):
    return [[
        get_img_tuple_fov_diff(f)[0],
        get_img_tuple_fov_diff(f)[1],
        get_img_tuple_fov_diff(f)[2],
        get_img_tuple_fov_diff(f)[3],
    ] for f in files]

def get_tuples_fov_diff_fv(files):
    return [[
        get_img_tuple_fov_diff_fv(f)[0],
        get_img_tuple_fov_diff_fv(f)[1],
        get_img_tuple_fov_diff_fv(f)[2],
        get_img_tuple_fov_diff_fv(f)[3],
    ] for f in files]


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


def get_img_tuple_fov_same(path):
    root = os.path.dirname(path)

    pair = Image.open(path)
    pair_basename = os.path.basename(path).split('_')

    fov_cat = pair_basename[-2][0]  # b
    fov_img_id = f'{fov_cat}{str(random.randint(1,30))}'  # b1
    fov_basename = pair_basename[:-2]
    fov_basename.extend([fov_img_id, fov_img_id + '.png'])
    fov_basename = '_'.join(fov_basename)
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


def get_img_tuple_fov_diff(path):
    root = os.path.dirname(path)

    pair = Image.open(path)
    pair_basename = os.path.basename(path).split('_')
    per_cat = pair_basename[-2][0]  # c

    cats = ['b', 'c', 'f', 'm']
    cats.remove(per_cat)

    fov_cat = cats[random.randint(0, 2)]  # b
    fov_img_id = f'{fov_cat}{str(random.randint(1,30))}'  # b1
    fov_basename = pair_basename[:-2]
    fov_basename.extend([fov_img_id, fov_img_id + '.png'])
    fov_basename = '_'.join(fov_basename)
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

def get_img_tuple_fov_diff_fv(path):
    root = os.path.dirname(path)

    pair = Image.open(path)
    pair_basename = os.path.basename(path).split('_')
    per_cat = pair_basename[-2][0]  # c

    cats = ['b', 'c', 'f', 'm']

    if (per_cat == 'b') or (per_cat == 'c'):
        cats.remove('b')
        cats.remove('c')
    else:
        cats.remove('f')
        cats.remove('m')


    fov_cat = cats[random.randint(0, 1)]  # b
    fov_img_id = f'{fov_cat}{str(random.randint(1,30))}'  # b1
    fov_basename = pair_basename[:-2]
    fov_basename.extend([fov_img_id, fov_img_id + '.png'])
    fov_basename = '_'.join(fov_basename)
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
        shuffle=True,
        device=defaults.device,
    )

    # check that train and test splits have balanced classes
    # train_test = ["TRAIN", "TEST"]
    # for train_test_id in [0, 1]:
    #     s = 0
    #     d = 0
    #     for item in dls.__getitem__(train_test_id).items:
    #         # print(label_from_path(item))
    #         # print(item)
    #         # print('---')
    #         if item[3] == 1:
    #             s += 1
    #         else:
    #             d += 1
    #     print(
    #         f"{train_test[train_test_id]} SET (same, diff): {str(s)}, {str(d)}"
    #     )

    return dls


def make_dls_fov_same(stim_path, batch_sz=24, seed=0, test_prop=0.2):

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
        get_items=get_tuples_fov_same,
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


def make_dls_fov_diff(stim_path, batch_sz=24, seed=0, test_prop=0.2):

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
        get_items=get_tuples_fov_diff,
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

def make_dls_fov_diff_fv(stim_path, batch_sz=24, seed=0, test_prop=0.2):

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
        get_items=get_tuples_fov_diff_fv,
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
    try:
        net.fb[0].weight.requires_grad = True
        net.fb[0].bias.requires_grad = True
    except:
        pass

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
                print("{0:0.2f}".format(cycle + 1),
                      "{0:0.2f}".format(epoch + 1),
                      "{0:0.2f}".format(tr_running_loss),
                      "{0:0.2f}".format(te_running_loss),
                      "{0:0.2f}".format(100 * tr_correct / tr_total),
                      "{0:0.2f}".format(100 * te_correct / te_total),
                      "{0:0.2f}".format(end))

    return (tr_loss, tr_acc, te_loss, te_acc, cf_pred, cf_y)


def test_net_fov_decode(net, p):

    cycles = p[0]
    epochs = p[1]
    train_loader = p[2]
    test_loader = p[3]
    noise_vars = p[4]

    X = []
    y = []
    res = []

    # TODO: Here, we need to assess if image category can be decoded via
    # something like MVPA from fov V1. We first need to extract X and y and
    # then pipe into a standard sklearn svm pipeline.

    # TODO: Does it make sense to revise the def of model 0 to include FB?
    net.fb.register_forward_hook(lambda m, input, output: print(output))

    net.eval()
    with torch.no_grad():
        for (inputs, labels) in train_loader:

            # TODO: get activation in v1 ROIs
            # TODO: With the hook above, I believe this will only print.
            # TODO: Revise from here...
            X = net(inputs)
            y = labels

    X = x
    y = y

    clf = make_pipeline(StandardScaler(), SVC())

    skf = StratifiedKFold(n_splits=5)

    f = 0
    for train_index, test_index in skf.split(X, y):
        f += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        d = pd.DataFrame({'fold': f, 'acc': acc})
        res.append(d)

    res = pd.concat(res)

    return res


def test_net_noise(net, p):

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
                          100 * te_correct / te_total, end)

        d = pd.DataFrame({
            'noise_var': v,
            'te_loss': te_loss,
            'te_acc': te_acc
        })
        res.append(d)

    res = pd.concat(res)

    return res


def test_net_fovimg(net, p):

    cycles = p[0]
    epochs = p[1]
    train_loader = p[2]
    test_loader = p[3]
    train_loader_same = p[4]
    test_loader_same = p[5]
    train_loader_diff = p[6]
    test_loader_diff = p[7]

    tl_list = [train_loader, train_loader_same, train_loader_diff]
    tl_names = ['base', 'same', 'diff']

    res = []

    criterion = nn.CrossEntropyLoss()

    for ii, tl in enumerate(tl_list):

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
                    for (inputs, labels) in tl:

                        fig, ax = plt.subplots(1, 3, squeeze=False)
                        fov_img = inputs[2].detach().cpu().numpy()
                        p1_img = inputs[0].detach().cpu().numpy()
                        p2_img = inputs[1].detach().cpu().numpy()
                        ax[0, 0].imshow(p1_img[0, 0, :, :])
                        ax[0, 1].imshow(p2_img[0, 0, :, :])
                        ax[0, 2].imshow(fov_img[0, 0, :, :])
                        plt.show()

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
                          100 * te_correct / te_total, end)

        d = pd.DataFrame({'net': net.module.model_name, 'tl': [tl_names[ii]], 'te_loss': te_loss, 'te_acc': te_acc})
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

    for r in res:

        tr_loss = r[0]
        tr_acc = r[1]
        te_loss = r[2]
        te_acc = r[3]
        cf_pred = r[4]
        cf_y = r[5]

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

    init_weights(net_0)
    params_to_update = net_0.parameters()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params_to_update),
                           lr=lr_min,
                           weight_decay=weight_decay)


    criterion = nn.CrossEntropyLoss()

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

    inspect_results(res)


def test_nets_noise(p):

    stim_path = p[0]
    epochs = p[1]
    cycles = p[2]
    batch_sz = p[3]
    lr_min = p[4]
    weight_decay = p[5]
    w_dropout_1 = p[6]
    w_dropout_2 = p[7]
    seed = p[8]

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

    dls = make_dls(stim_path, batch_sz, seed)
    train_loader = dls.train
    test_loader = dls.valid
    # dls.show_batch(max_n = 2)
    # plt.show()

    cycles = 1
    epochs = 1
    noise_levels = np.linspace(0.0,2.0, 10)
    batch_sz = test_loader.n

    p = (cycles, epochs, train_loader, test_loader, noise_levels)

    res = [test_net_noise(x, p) for x in nets]

    return res


def test_nets_fovimg(p):

    stim_path = p[0]
    epochs = p[1]
    cycles = p[2]
    batch_sz = p[3]
    lr_min = p[4]
    weight_decay = p[5]
    w_dropout_1 = p[6]
    w_dropout_2 = p[7]
    seed = p[8]

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

    dls = make_dls(stim_path, batch_sz, seed)
    train_loader = dls.train
    test_loader = dls.valid
    # dls.show_batch(max_n = 2)
    # plt.show()

    dls_same = make_dls_fov_same(stim_path, batch_sz, seed)
    train_loader_same = dls_same.train
    test_loader_same = dls_same.valid

    dls_diff = make_dls_fov_diff(stim_path, batch_sz, seed)
    train_loader_diff = dls_diff.train
    test_loader_diff = dls_diff.valid

    cycles = 1
    epochs = 1
    noise_levels = np.linspace(0.0, 3.0, 100)
    batch_sz = test_loader.n

    p = (cycles, epochs, train_loader, test_loader, train_loader_same,
         test_loader_same, train_loader_diff, test_loader_diff)

    res = [test_net_fovimg(x, p) for x in nets]

    return res


def test_nets_fov_decode(p):

    stim_path = p[0]
    epochs = p[1]
    cycles = p[2]
    batch_sz = p[3]
    lr_min = p[4]
    weight_decay = p[5]
    w_dropout_1 = p[6]
    w_dropout_2 = p[7]
    seed = p[8]

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

    dls = make_dls(stim_path, batch_sz, seed)
    train_loader = dls.train
    test_loader = dls.valid
    # dls.show_batch(max_n = 2)
    # plt.show()

    cycles = 1
    epochs = 1
    noise_levels = np.linspace(0.0, 3.0, 100)
    batch_sz = test_loader.n

    p = (cycles, epochs, train_loader, test_loader)

    res = [test_net_fov_decode(x, p) for x in nets]

    return res
