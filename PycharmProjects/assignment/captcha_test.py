import torch
from torch.autograd import Variable
from DataSet import dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

LR=0.001
BATCH_SIZE=16

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

state_dict=torch.load('epoch2.pkl')
net=CNN()
net.load_state_dict(state_dict).cuda()
test_data=dataset('./test','./test_label.txt',transform=transforms.ToTensor())
test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,drop_last=True)
test_data_size=len(test_data)

loss_func = nCrossEntropyLoss()

for input, label in test_dataloader:
    pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
    input = Variable(input).cuda()
    label = Variable(label).cuda()
    output = net(input)
    loss = loss_func(output, label)
    for i in rangecat(4):
        pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)
        pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)

    with open('result.txt','a') as f:
        f.write(pred[:,1:])
        f.writelines('\n')


