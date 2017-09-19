#!/usr/bin/env python


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

from loadCOCO import loadCOCO


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv64 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv512 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.upconv1024 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dconv1024 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv512 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dconv512 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv256 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv256 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv128 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv128 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 182, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = F.relu(self.conv64(x))
        x2 = F.relu(self.conv128(self.pool(x1)))
        x3 = F.relu(self.conv256(self.pool(x2)))
        x4 = F.relu(self.conv512(self.pool(x3)))
        x5 = F.relu(self.conv1024(self.pool(x4)))
        ux5 = self.upconv1024(x5)
        cc5 = torch.cat([ux5, x4], 1)
        dx4 = F.relu(self.dconv1024(cc5))
        ux4 = self.upconv512(dx4)
        cc4 = torch.cat([ux4, x3], 1)
        dx3 = F.relu(self.dconv512(cc4))
        ux3 = self.upconv256(dx3)
        cc3 = torch.cat([ux3, x2], 1)
        dx2 = F.relu(self.dconv256(cc3))
        ux2 = self.upconv128(dx2)
        cc2 = torch.cat([ux2, x1], 1)
        dx1 = F.relu(self.dconv128(cc2))  # no relu?
        last = self.conv1(dx1)
        return F.log_softmax(last)  # sigmoid if classes arent mutually exclusv

###########
# Load Dataset  #
###########
ims, labs = loadCOCO("/home/toni/Data/COCOstuff/")
imsT = torch.Tensor(ims)
labsT = torch.ByteTensor(labs)
trainset = TensorDataset(imsT, labsT)
trainloader = torch.utils.data.DataLoader(
                                                                trainset,
                                                                batch_size=4,
                                                                shuffle=True,
                                                                num_workers=2
                                                                )


net = Net()
criterion = nn.NLLLoss2d()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
