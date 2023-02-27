import torch
import torch.nn.functional as F
import numpy as np

from torchsummary import summary
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


class FastGAN(nn.Module):
    """ TOWARDS FASTER AND STABILIZED GAN TRAINING FOR HIGH-FIDELITY FEW-SHOT IMAGE SYNTHESIS

    Training Generatice Adversarial Networks (GAN)
    this model try to develop the few-shot image 
    synthesis task for GAN with minimum computing
    cost.

    With this model we expect get a model convergency 
    from scratch in less than a pair of hours of 
    traning on a sigle RTX-2080. [Modelo por definir]

    Also we include a pre-trained model using [cita]
    and data aumentation to generate more examples.
    """
    def __init__(self):
        super(FastGAN, self).__init__()


    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, filters=128, noise_dim=64):
        super(Generator, self).__init__()
        self.filters = filters
        self.init = nn.Sequential(
            nn.Linear(noise_dim, filters * 4 * 4),
            nn.BatchNorm1d(filters * 4 * 4),
            nn.LeakyReLU(0.2),
        )             
        
        self.feat_8   = upBlock(filters, filters)
        self.feat_16  = upBlock(filters, filters // 2)
        self.feat_32  = upBlock(filters // 2, filters // 4)
        
        self.ch_conv = nn.Conv2d(filters // 4, 3, 3, 1, 1, bias=False)

    def forward(self, z):
        feat_4 = self.init(z)
        feat_4 = torch.reshape(feat_4, (-1, self.filters, 4, 4))
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        img = torch.tanh(self.ch_conv(feat_32))
        return img
    
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2)
    )
    return block