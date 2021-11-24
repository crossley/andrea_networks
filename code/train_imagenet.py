# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:31:08 2021

@author: mq20185996
"""
import os
import torch
import torch.utils.model_zoo
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from pathlib import Path
import collections

from fastai.vision.all import *
from kornia import rgb_to_grayscale


if __name__ == '__amain__':

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    data_dir = Path(r'D:\Andrea_NN\data\IMAGENET\Compressed')
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'validation')
    
    train_dataset = datasets.ImageNet(
        traindir,
        split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_dataset = datasets.ImageNet(
        valdir,
        split='val',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))    
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=256,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)
    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
    
            # V1 layers
            self.V1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2,
                          padding=7 // 2),  # + self.vfb,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # V2 layers
            self.V2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # V4 layers
            self.V4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # IT layers
            self.IT = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # feedback layers (v4 to v1)
            self.FB = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=3 // 2),
                nn.AdaptiveAvgPool2d(112),
            )
    
            # decoding layer
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(512, 1000))
    
        def forward(self, inp):
    
            v4 = torch.zeros(512, 256, 3, 3).to('cuda')
            vIT = torch.zeros(256, 128, 3, 3).to('cuda')
            v2 = torch.zeros(256, 128, 3, 3).to('cuda')
            v1 = torch.zeros(128, 64, 3, 3).to('cuda')
    
            T = 4
            for t in range(T):
                # NOTE: By running through the network in reverse order, we are
                # assuming that it takes one forward pass for activation in each
                # area to be propagated forward.
                if t == 0:
                    x = inp
                else:
                    x = torch.zeros_like(inp)
    
                vIT = self.IT(v4)
                # vfb = self.FB(v4)
                v4 = self.V4(v2)
                v2 = self.V2(v1)
                v1 = self.V1(x)
    
            # use this output for training the feedback weights
            # out = v1
    
            # use this output for training same / different judgements
            # out = nn.AdaptiveAvgPool2d(1)(vIT)
            # out = out.view(out.size(0), -1)
            # out = self.decoder(out)
    
            #use this for multi-classification (training)
            out = self.decoder(vIT)
    
            return out
        
    class Debug_Net(nn.Module):
        def __init__(self):
            super(Debug_Net, self).__init__()
    
            # V1 layers
            self.V1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2,
                          padding=7 // 2),  # + self.vfb,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # V2 layers
            self.V2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # V4 layers
            self.V4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # IT layers
            self.IT = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
            # decoding layer
            self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(512, 1000))
    
        def forward(self, inp):
            
            v4 = torch.zeros(512, 256, 3, 3).to('cuda')
            vIT = torch.zeros(256, 128, 3, 3).to('cuda')
            v2 = torch.zeros(256, 128, 3, 3).to('cuda')
            v1 = torch.zeros(128, 64, 3, 3).to('cuda')
    
            v1 = self.V1(inp)
            v2 = self.V2(v1)
            v4 = self.V4(v2)
            vIT = self.IT(v4)
            out = self.decoder(vIT)
    
            return out
    
    # net = Debug_Net()
    net = Net()

    
    # turn off gradient tracking for static parameters
    # for param in net.FB.parameters():
    #     param.requires_grad = False
    
    # weight initialization
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        
    PATH = Path(r'C:\Users\mq20185996\Documents\Python Scripts\net.pt')
    # PATH = Path(r'C:\Users\mq20185996\Documents\Python Scripts\debug_net.pt')
    # tmp = torch.load(PATH)
    
    # PATH = Path(r'C:\Users\mq20185996\Documents\Python Scripts\cornet_z-5c427c9c.pth')
    if PATH.is_file():
        ckpt_data = torch.load(PATH)
        state_dict = ckpt_data
        # state_dict = ckpt_data['state_dict']
        # state_dict = collections.OrderedDict([
        #     ('V1.0.weight', state_dict['module.V1.conv.weight']),
        #     ('V1.0.bias', state_dict['module.V1.conv.bias']),
        #     ('V2.0.weight', state_dict['module.V2.conv.weight']),
        #     ('V2.0.bias', state_dict['module.V2.conv.bias']),
        #     ('V4.0.weight', state_dict['module.V4.conv.weight']),
        #     ('V4.0.bias', state_dict['module.V4.conv.bias']),
        #     ('IT.0.weight', state_dict['module.IT.conv.weight']),
        #     ('IT.0.bias', state_dict['module.IT.conv.bias']),
        #     ('decoder.2.weight', state_dict['module.decoder.linear.weight']),
        #     ('decoder.2.bias', state_dict['module.decoder.linear.bias']),
        #     ])
        net.load_state_dict(state_dict)
        # net.load_state_dict(torch.load(PATH))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)
    net.to(device)
    
    # train the network
    # net.train()
    # params_to_update = net.parameters()
    # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    
    # for epoch in range(5):
    #     running_loss = 0.0
    #     for i, (inputs, labels) in enumerate(train_loader):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    
    #     print(i)
    #     print('%5d loss: %.3f' % (epoch + 1, running_loss))
    #     running_loss = 0.0
        
    #     torch.save(net.state_dict(), PATH)
    
    # print('Finished Training')
    
    # test performance
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on all test images: %d %%' %
          (100 * correct / total))