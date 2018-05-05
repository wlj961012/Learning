import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake dataset 假的数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataset
#plt.scatter(x.numpy(), y.numpy())
#plt.show()

# 使用上节内容提到的 data loader 批训练
torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

class Net(torch.nn.Module):

    def __init__(self):

        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()

nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]
opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]
loss_func=torch.nn.MSELoss()

losses_his=[[],[],[],[]]

for epoch in range(EPOCH):
    for step,(batchx,batchy) in enumerate(loader):
        b_x=Variable(batchx)
        b_y=Variable(batchy)

        for net,opt,l_h in zip(nets,optimizers,losses_his):
            out=net(b_x)
            loss=loss_func(out,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_h.append(loss.data[0])


for l_s in losses_his:
    plt.plot(l_s)
plt.show()