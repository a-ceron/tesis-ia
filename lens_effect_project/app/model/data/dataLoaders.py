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

class DataLoaderFactory:
    def get_cifar10(root, download, batch_size, transform):
        test = get_cifar10_dataloader(
            root, 
            False,
            download,
            batch_size,
            transform
        )
        train = get_cifar10_dataloader(
            root, 
            True,
            download,
            batch_size,
            transform
        )
        return train, test
    

    def get_stl10(root, download, batch_size, transform):
        test = get_stl10_dataloader(
            root, 
            False,
            download,
            batch_size,
            transform
        )
        train = get_stl10_dataloader(
            root, 
            True,
            download,
            batch_size,
            transform
        )
        return train, test

    def get_galaxy():
        pass

    def get_galaxy_lens():
        pass
####################
def get_cifar10_dataset(root, train, download, transform):
    return CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )

def get_cifar10_dataloader(root, train, download, batch_size, transform):
    return DataLoader(
        get_cifar10_dataset(root, train, download, transform),
        batch_size=batch_size,
        shuffle=True
    )

def get_stl10_dataset(root, train, download, transform):
    return STL10(
        root=root,
        split='train' if train else 'test',
        download=download,
        transform=transform
    )

def get_stl10_dataloader(root, train, download, batch_size, transform):
    return DataLoader(
        get_stl10_dataset(root, train, download, transform),
        batch_size=batch_size,
        shuffle=True
    )