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

def make_dls_abstract(root, batch_sz=24, seed=0, test_prop=0.2):

    root = Path(root)
    cats = ['Cubies', 'Smoothies', 'Spikies']

    dirs = [os.path.join(root, cat) for cat in cats]
    stims = {
        os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
        for dir_ in dirs
    }
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [
        stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])
    ]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    def calc_dist(pair):
        dist = 0
        file1 = os.path.basename(pair[0]).split('_')[1][:4]
        file2 = os.path.basename(pair[1]).split('_')[1][:4]
        for i in range(4):
            dist += abs(int(file1[i]) - int(file2[i]))
        return dist

    fnames = [[fnames1[i], fnames2[i]] for i in range(len(fnames1))
              if calc_dist(pair) > 5]

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
    stims = {
        os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
        for dir_ in dirs
    }
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [
        stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])
    ]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    fnames = [[fnames1[i], fnames2[i]] for i in range(len(fnames1))]

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
    stims = {
        os.path.split(dir_)[1]: glob.glob(os.path.join(dir_, "*.jpg"))
        for dir_ in dirs
    }
    n_stims = len(stims[cats[1]])

    fnames1 = [stims[cat][i] for cat in stims for i in range(n_stims)] * 2

    def shuffle_list(list_):
        idxs1 = list(range(len(list_)))
        idxs2 = list(range(len(list_)))
        while sum([idxs1[i] == idxs2[i] for i in range(len(idxs1))]) > 0:
            random.shuffle(idxs1)
        return idxs1

    fnames2 = [stims[cat][i] for cat in stims for i in range(n_stims)] + [
        stims[cat][i] for cat in stims for i in shuffle_list(stims[cat])
    ]
    y = [fnames1[i] == fnames2[i] for i in range(len(fnames1))]

    print(f'SAME: {sum(y)}\nDIFF: {len(y)-sum(y)}')

    fnames = [[fnames1[i], fnames2[i]] for i in range(len(fnames1))]

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
