"""
###########################################
Esta es una implementación de clase de carga
de datos.

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""

from torchvision.datasets import CIFAR10, STL10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

####################
class DataLoaderLabels:
    cifar10 = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    stl10 = {
        0: 'airplane',
        1: 'bird',
        2: 'car',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'horse',
        7: 'monkey',
        8: 'ship',
        9: 'truck'
    }


####################
def get_cifar10_dataset(root, train, download):
    return CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=ToTensor()
    )

def get_cifar10_dataloader(root, train, download, batch_size):
    return DataLoader(
        get_cifar10_dataset(root, train, download),
        batch_size=batch_size,
        shuffle=True
    )

def get_stl10_dataset(root, train, download):
    return STL10(
        root=root,
        split='train' if train else 'test',
        download=download,
        transform=ToTensor()
    )