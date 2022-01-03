# -*- coding: utf-8 -*-
"""
Created on Sat Mon 03 16:00:00 2022

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

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # TODO: train imagenet (but in separate file)
    # data_dir = Path(r'D:\Andrea_NN\data\IMAGENET\Compressed')
    # traindir = os.path.join(data_dir, 'train')
    # valdir = os.path.join(data_dir, 'validation')

    # train_dataset = datasets.ImageNet(traindir,
    #                                   split='train',
    #                                   transform=transforms.Compose([
    #                                       transforms.RandomResizedCrop(224),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       normalize,
    #                                   ]))

    # val_dataset = datasets.ImageNet(valdir,
    #                                 split='val',
    #                                 transform=transforms.Compose([
    #                                     transforms.RandomResizedCrop(224),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                     normalize,
    #                                 ]))

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=256,
    #                                            shuffle=True,
    #                                            num_workers=4,
    #                                            pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=256,
    #                                          shuffle=False,
    #                                          num_workers=4,
    #                                          pin_memory=True)
