import torch
import torch.nn.functional as F

from torchsummary import summary
from numpy import power
from torch import nn
from collections import OrderedDict

class cnn_block(nn.Module):
    def __init__(self):
        super(cnn_block, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)    
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        return x

class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()

        self.block = cnn_block()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*2*2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

