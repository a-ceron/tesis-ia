import const
import trainer as T
import generator as G

from models.dataset import Dataset_Lens
from torch.utils.data import DataLoader
from torchvision import transforms
from numpy import pi

def lens_dataloader(label:str, batch_size:int, shuffle:bool, 
                    num_workers:int, pin_memory:bool) -> DataLoader:
    """Regresa un dataloader para el conjunto de datos

    in:
        label:str
            Nombre de la carpeta cotenedora
        **kwargs
            Otros datos
    out:
        Dataloader
    """
    if label == 'train':
        transformer = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(),
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(pi),
                # transforms.RandomInvert(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomEqualize()
            ]),
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
    dataset = Dataset_Lens(const.PATH_OF_FIGURES, label, transformer)

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=pin_memory)
    return dataloader

def main():
    label = 'test'
    batch_size = const.BATCH_SIZE_32
    num_workers = const.TWO_WORKERS
    pin_memory = const.TRUE_VALUE
    shuffle = const.TRUE_VALUE

    dataloader = lens_dataloader(label, batch_size, shuffle, num_workers, pin_memory)

    print(len(dataloader.dataset))

if __name__ == '__main__':
    main()