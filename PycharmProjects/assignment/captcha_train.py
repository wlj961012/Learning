from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.functional
from DataSet import dataset
import torch.nn.functional as F
import numpy as np

EPOCH=10
BATCH_SIZE=8
LR = 0.001


data = dataset('./src','./label.txt', transform=transforms.ToTensor())
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
dataset_size = len(data)

test_data=dataset('./test','./test_label.txt',transform=transforms.ToTensor())
test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,drop_last=True)
test_data_size=len(test_data)

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2,),#3*160*60
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#16*80*30
            nn.Conv2d(16,32,5,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#32*40*15
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#64*20*7
        )
        self.out1=nn.Linear(64*20*7,500)
        self.out2=nn.Linear(500,40)

    def forward(self, x):
        x=self.conv(x)
        x = x.view(x.size(0), -1)
        x=self.out1(x)
        out=self.out2(x)
        return out


class nCrossEntropyLoss(torch.nn.Module):

    def __init__(self,n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self,output,label):
        self.total_loss = 0
        label=label.long()
        for i in range(self.n):
            output_t=output[:,10*i:10*i+10]
            label_t=label[:,i]
            self.total_loss += self.loss(output_t, label_t)
        return self.total_loss

cnn=CNN().cuda()
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nCrossEntropyLoss()

best_acc=0

for epoch in range(EPOCH):

    running_loss = 0.0
    running_corrects = 0

    for step,(inputs,label) in enumerate(dataloader):

        pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
        inputs = Variable(inputs).cuda()
        label = Variable(label).cuda()
        optimizer.zero_grad()
        output = cnn(inputs)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        for i in range(4):
            pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)
            pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)

        running_loss += loss.data[0] * inputs.size()[0]
        running_corrects += sum(pred.numpy()[:, 1:]==label.data.cpu().numpy()).astype(int)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    epoch_acc=np.sum(epoch_acc)


    test_loss=0
    test_corrects=0

    for input,label in test_dataloader:
        pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
        input=Variable(input).cuda()
        label=Variable(label).cuda()
        output=cnn(input).cuda()
        loss = loss_func(output, label)
        for i in range(4):
            pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)
            pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)
        test_loss += loss.data[0] * input.size()[0]
        test_corrects += sum(pred.numpy()[:, 1:]==label.data.cpu().numpy()).astype(int)

    test_loss=test_loss/test_data_size
    test_acc=test_corrects/test_data_size
    test_acc=np.sum(test_acc)

    if test_acc>best_acc:
        best_acc = epoch_acc
        torch.save(cnn.state_dict(),"epoch%d.pkl"%epoch)

    print("loss", epoch_loss, 'acc', epoch_acc/4.,'testloss',test_loss,'test_acc',test_acc/4.)





