import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH=1
BATCH_SIZE=50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)

#批训练数据
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]

#print(train_data.data.size)


class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        #x=self.conv2(x)
        
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output

cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
print(cnn)
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):

        b_x=Variable(x)
        b_y=Variable(y)

        output=cnn(x)
        loss=loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if step==50:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.squeeze()
            print(loss.data[0])
