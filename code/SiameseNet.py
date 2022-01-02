# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:10:51 2021

@author: mq20185996
"""

from imports import *

# class CORblock_Z(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels,
#                               out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=kernel_size // 2)
#         self.nonlin = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.output = Identity()

#     def forward(self, inp):
#         x = self.conv(inp)
#         x = self.nonlin(x)
#         x = self.pool(x)
#         x = self.output(x)
#         return x


class SiameseNet(nn.Module):
    def __init__(self, w_dropout_1, w_dropout_2, head_mult):
        super(SiameseNet, self).__init__()

        self.head_mult = head_mult

        self.feature_map_V1_fov = torch.empty((24, 64, 56, 56))
        self.feature_map_fb = torch.empty((24, 61, 224, 224))
        self.feature_map_V1 = torch.empty((24, 64, 56, 56))
        self.feature_map_V2 = torch.empty((24, 64, 28, 28))
        self.feature_map_V4 = torch.empty((24, 64, 14, 14))
        self.feature_map_IT = torch.empty((24, 64, 7, 7))

        self.V1_fov = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.IT = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
        # self.V2 = CORblock_Z(64, 128)
        # self.V4 = CORblock_Z(128, 256)
        # self.IT = CORblock_Z(256, 512)

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(1024 * self.head_mult,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Dropout(p=w_dropout_1, inplace=False),
            nn.Linear(in_features=1024 * self.head_mult,
                      out_features=512,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.Dropout(p=w_dropout_2, inplace=False),
            nn.Linear(in_features=512, out_features=2, bias=False),
        )

    def forward(self, inp):
        pass

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_pretrained_weights(self):
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

        self.load_state_dict(state_dict, strict=False)

    def freeze_pretrained_weights(self):
        self.V1[0].weight.requires_grad = False
        self.V1[0].bias.requires_grad = False
        self.V2[0].weight.requires_grad = False
        self.V2[0].bias.requires_grad = False
        self.V4[0].weight.requires_grad = False
        self.V4[0].bias.requires_grad = False
        self.IT[0].weight.requires_grad = False
        self.IT[0].bias.requires_grad = False

    def train_net(self, optimizer, criterion, data_loader, cycles, epochs):

        net = self
        train_loader = data_loader.train
        test_loader = data_loader.valid

        stop_train_crit = 75.0

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axs = plt.subplots(ncols=len(imgs),
                                    squeeze=False,
                                    figsize=(10, 10))
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[],
                              yticklabels=[],
                              xticks=[],
                              yticks=[])

        net_layer = self.fb
        net_layer_name = 'fb'
        handle = net_layer[0].register_forward_hook(
            get_activation(net_layer_name))

        for cycle in range(cycles):
            tr_loss = []
            tr_acc = []
            te_loss = []
            te_acc = []

            for epoch in range(epochs):
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

                net.eval()
                te_running_loss = 0.0
                te_correct = 0
                te_total = 0
                cf_pred = []
                cf_y = []
                with torch.no_grad():
                    for (inputs, labels) in test_loader:
                        out = net(inputs)
                        # act_fb = activation['fb']
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
                    # act_fb = torch.unbind(act_fb[0,:,:,:], 0)
                    # act_fb = [x for x in act_fb]
                    # grid = torchvision.utils.make_grid(act_fb, padding=10)
                    # show(grid)

            #     if te_acc[-1] >= stop_train_crit:
            #         break
            # if te_acc[-1] >= stop_train_crit:
            #     break

        return (tr_loss, tr_acc, te_loss, te_acc, cf_pred, cf_y)

    def test_net(self, criterion, test_loader):

        net = self
        net.eval()

        te_running_loss = 0.0
        te_correct = 0
        te_total = 0
        te_loss = []
        te_acc = []
        cf_pred = []
        cf_y = []
        with torch.no_grad():
            start = time.time()
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
            print("{0:0.2f}".format(te_running_loss),
                  "{0:0.2f}".format(100 * te_correct / te_total),
                  "{0:0.2f}".format(end))

        return (te_loss, te_acc, cf_pred, cf_y)
