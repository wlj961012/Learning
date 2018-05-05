import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
# import pandas as pd
import numpy as np
import os
import copy, time

file_path = ''
BATCH_SIZE = 16
EPOCH = 10

class dataset(Dataset):

    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label = np.loadtxt(label_file)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '%s.jpg' % idx)
        image = Image.open(img_name)
        labels = self.label[idx, :]
        if self.transform:
            image = self.transform(image)
        return image,labels

    def __len__(self):
        return (self.label.shape[0])


data = dataset(file_path + './src', file_path + './label.txt', transform=transforms.ToTensor())
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
dataset_size = len(data)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # in:(bs,3,60,160)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(64 * 7 * 20, 500)
        self.fc2 = nn.Linear(500, 40)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # reshape to (batch_size, 64 * 7 * 30)
        output = self.fc1(x)
        output = self.fc2(output)
        return output

class nCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output_t = output[:, 0:10]
        label = Variable(torch.LongTensor(label.data.cpu().numpy()))
        label_t = label[:, 0]
        for i in range(1, self.n):

            output_t = torch.cat((output_t, output[:, 10 * i:10 * i + 10]),0)
            label_t = torch.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss


def equal(np1, np2):
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i, :] == np2[i, :]).all():
            n += 1

    return n


net = ConvNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nCrossEntropyLoss()

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

since = time.time()
for epoch in range(EPOCH):

    running_loss = 0.0
    running_corrects = 0

    for step, (inputs, label) in enumerate(dataloader):

        pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
        inputs = Variable(inputs)
        label = Variable(label)
        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_func(output, label)

        for i in range(4):
            pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)
            pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * inputs.size()[0]
        running_corrects += equal(pred.numpy()[:, 1:], label.data.cpu().numpy().astype(int))


    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    print("loss",epoch_loss,'acc',epoch_acc)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Train Loss:{:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
