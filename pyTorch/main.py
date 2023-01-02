from models import models

import torchvision
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def CNN():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size = 32
    K = len(classes)
    # Paso 1. Obtenemos los datos
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True,
                                        download=True)
    trainloader = torch.utils.data.DataLoader(trainset,                 
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',   
                                        train=False,
                                        download=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    # Paso 2. Creamos el modelo
    cnn = models.MyCNN(K)

    print(cnn)
    # Paso 3. Entrenamos el modelo

def main():
    CNN()

if __name__ == '__main__':
    main()