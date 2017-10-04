#!/usr/bin/env python

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from scipy.misc import imshow
from tqdm import tqdm

from loadCOCO import loadCOCO, Rescale, RandomCrop


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
        self.conv1 = nn.Conv2d(64, 183, 1)
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


def save_checkpoint(model, epoch, iteration, loss, vloss):
        checkpoint = {}
        checkpoint["model"] = model
        checkpoint["epoch"] = epoch
        checkpoint["iteration"] = iteration
        checkpoint["loss"] = loss
        checkpoint["vloss"] = vloss
        fname = "checkpoint_" + str(epoch) + "_" + str(iteration) + ".dat"
        torch.save(checkpoint, fname)
        return


def train(resume_from=None):
        ###########
        # Load Dataset  #
        ###########
        ims, labs = loadCOCO("/home/toni/Data/COCOstuff/")
        imsTrain = ims[0:int(0.95*len(ims))]
        labsTrain = labs[0:int(0.95*len(labs))]
        imsValid = ims[int(0.95*len(ims)):]
        labsValid = labs[int(0.95*len(labs)):]

        imsTrainT = torch.Tensor(imsTrain)
        labsTrainT = torch.ByteTensor(labsTrain)
        imsValidT = torch.Tensor(imsValid)
        labsValidT = torch.ByteTensor(labsValid)
        trainset = TensorDataset(imsTrainT, labsTrainT)
        validset = TensorDataset(imsValidT, labsValidT)
        trainloader = torch.utils.data.DataLoader(
                                                                trainset,
                                                                batch_size=1,
                                                                shuffle=True,
                                                                num_workers=2
                                                                )

        validloader = torch.utils.data.DataLoader(
                                                                validset,
                                                                batch_size=1,
                                                                shuffle=True,
                                                                num_workers=2
                                                                )

        net = Net()
        if torch.cuda.is_available():
                net.cuda()

        criterion = nn.NLLLoss2d()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.005)

        if resume_from is not None:
            checkpoint = torch.load(resume_from)

        checkpoint_rate = 500
        for epoch in range(12):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, start=0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda()),\
                                                 Variable(labels.cuda())
                else:
                        inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % checkpoint_rate == checkpoint_rate-1:    # print every N mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint_rate))

                    # Validation test
                    running_valid_loss = 0.0
                    running_valid_acc = 0.0
                    for j, data in enumerate(validloader, 0):
                        inputs, labels = data

                        # wrap them in Variable
                        if torch.cuda.is_available():
                                inputs, labels = Variable(inputs.cuda()),\
                                                        Variable(labels.cuda())
                        else:
                                inputs, labels = Variable(inputs), Variable(labels)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()
                        # print statistics
                        running_valid_loss += loss.data[0]
                        running_valid_acc +=  \
                            ((outputs.max(1)[1] == labels.long()).sum()).float() \
                            / (labels.size()[1] * labels.size()[2])

                    print('[Validation loss]: %.3f' %
                          (running_valid_loss / len(imsValid)))

                    print('[Validation accuracy]: %.3f' %
                          ((running_valid_acc / len(imsValid)) * 100.0).data[0])

                    save_checkpoint(
                            net.state_dict(),
                            epoch+1,
                            i + 1,
                            running_loss / checkpoint_rate,
                            running_valid_loss / len(imsValid))
                    running_loss = 0.0

        print('Finished Training')


def test_image(paramsPath, img, label=None, showim=False):
        resc = Rescale(500)
        crop = RandomCrop(480)

        im, lbl = resc(img, label)
        im, lbl = crop(im, lbl)
        im = np.transpose(im, (2, 0, 1))
        im = np.array(im, dtype='float32')
        im /= 255.0
        im = (im*2)-1
        im = np.expand_dims(im, axis=0)
        imT = torch.Tensor(im)
        labT = torch.ByteTensor(lbl)
        imV, labV = Variable(imT), Variable(labT)

        net = Net()
        if torch.cuda.is_available():
                net.cuda()

        par = torch.load(paramsPath, map_location=lambda storage, loc: storage)
        net.load_state_dict(par["model"])

        if torch.cuda.is_available():
            out = net(imV.cuda())
            ouim = out.data.cpu()
        else:
            out = net(imV)
            ouim = out.data
        ouim = ouim.numpy()

        if showim:
            imshow(ouim[0])

        return ouim, lbl
        