# Author:StevenChaoo
# -*- coding:UTF-8 -*-


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt


class CNN_NET(torch.nn.Module):
    '''
    Create a CNN network

    Var:
        conv1: input is 3;
               output is 64;
               kernel is 5*5;
               step length is 1;
               padding is 0
        conv2: kernel is 3*3;
               step length is 2
        pool:  use Max-Pooling method
        fc1:   connect 64*4*4 and 384 as first layer
        fc2:   connect 384 and 192 as second layer
        fc3:   connect 192 and 10 as last layer
    Return:
        x:     activation function is ReLU
    '''

    def __init__(self):
        super(CNN_NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1,
                                     padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=3,
                                       stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 5)
        self.fc1 = torch.nn.Linear(64*4*4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def getData(BATCH_SIZE):
    '''
    Download data of CIFAR10 from internet or use existing data. Transform them
    to torch type

    Param:
        BATCH_SIZE: how much big should program take sample
    return:
        trainloader: load trainset which consists of 50,000 datapoints and split
                     as Train type
        testset:     load testset which consists of 10,000 datapoints and split
                     as Test type
    '''
    transform = trans.Compose([trans.ToTensor(),
                               trans.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='./data',
                                   train=True,
                                   download=False,
                                   transform=transform)
    testset = tv.datasets.CIFAR10(root='./data',
                                  train=False,
                                  download=False,
                                  transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    return trainloader, testloader


def testDataLoading():
    '''
    Test if data loading correctly
    '''
    transform = trans.Compose([trans.ToTensor(),
                               trans.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='./data',
                                   train=True,
                                   download=False,
                                   transform=transform)
    testset = tv.datasets.CIFAR10(root='./data',
                                  train=False,
                                  download=False,
                                  transform=transform)
    plt.imshow(trainset.data[77])
    # plt.imshow(testset.data[55])
    plt.show()


def trainData(EPOCH, net, trainloader):
    '''
    Train model with Stochastic gradient descent optimaizer and cross entropy
    loss function

    Param:
        EPOCH:       how many epoches will program excuate
        net:         CNN network
        trainloader: train data
    '''
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    print('\n----------Start Training----------\n')

    for epoch in range(EPOCH):
        running_loss = 0.0
        print('EPOCH = %d' % (epoch + 1))
        for step, data in enumerate(trainloader):
            b_x, b_y = data
            outputs = net.forward(b_x)
            loss = loss_func(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 1000 == 999:
                print('     Progress = %d%c\tLOSS = %.4f' %
                      ((step + 1) / 10000 * 100, '%', running_loss / 2000))
                running_loss = 0.0

    print('\n----------Finished Training----------')


def evaluate(testloader, net):
    '''
    Evaluate total loss

    Param:
        testloader: test data
        net:        trained CNN network
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = net(images)
            numbers, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the 10000 test images: %d %%\n' %
          (100 * correct / total))


if __name__ == "__main__":

    # Hyper-parameters
    BATCH_SIZE = 5
    EPOCH = 2

    # Initialize class CNN_NET()
    net = CNN_NET()

    # Core
    trainloader, testloader = getData(BATCH_SIZE)
    trainData(EPOCH, net, trainloader)
    evaluate(testloader, net)
