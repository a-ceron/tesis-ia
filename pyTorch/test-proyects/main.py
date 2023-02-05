import trainer as T
import generator as G
from models import const, tools, models
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchsummary import summary
from numpy import pi

def dowload_figures():
    df = tools.get_figures(const.PATH_OF_FIGURES, train = const.TRAIN, source=const.PATH_OF_TSV)

def lens_dataset():
    # Iniciamos construyendo algunas transformaciones 
    # predefinadas, en el futuro esperamos hacer las
    # propias
    transformador = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(),
            transforms.RandomGrayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(pi),
            # transforms.RandomInvert(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomEqualize()
        ]),
        transforms.ToTensor(),
        
    ])

    dataset = ImageFolder(const.PATH_OF_FIGURES, transform=transformador)
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE_128, shuffle=True, num_workers=4)
    return dataloader

def main():
    # dowload_figures()
    dataloader = lens_dataset()

    noise_dim = 64
    noise = torch.randn(const.BATCH_SIZE_128, noise_dim)
    generator = models.Generator(128, noise_dim)
    gen_batch = generator(noise)
    gen_batch.shape

    print(summary(generator, input_size=(1,128,128)))

if __name__ == '__main__':
    main()