"""
###########################################
Esta es una implementación de una arquitecura
CNN



autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""

from torch import nn

# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=32, 
            kernel_size=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3
        )
        self.max_pool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3
        )
        self.max_pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_pool1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out