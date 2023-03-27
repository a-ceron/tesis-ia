"""
###########################################
Esta es una implementación de una arquitecura
CNN

Este es un modelo que sirve para clasificar
imágenes. 

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from torch import nn

class ariCNN(nn.Module):
    """Usando ImageNet como referencia"""
    def __init__(self, num_classes):
        super(ariCNN, self).__init__()
        self.model = nn.Sequential(
            self._block(3, 32, 5, 1, 1),
            self._block(32, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def _block(self, 
        in_channels:int,
        out_channels:int,
        kernel_size:int or tuple, 
        stride: int or tuple, 
        padding: int or tuple):
        """A block of layers for the CNN

        This is trognly recommended as stackin multiple convolutional layers allows for more complex features of the input vector to be selected.

        :param in_channels: _description_
        :type in_channels: int
        :param out_channels: _description_
        :type out_channels: int
        :param kernel_size: _description_
        :type kernel_size: intortuple
        :param stride: _description_
        :type stride: intortuple
        :param padding: _description_
        :type padding: intortuple
        :return: _description_
        :rtype: _type_
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, X):
        return self.model(X)