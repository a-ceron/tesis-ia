import torchvision
from torchvision import transforms
from model.GANs import FastGAN
from model.data import dataManipulator
from model.utils import const, tools
import trainer as trainer


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def read_data():
    """Regresa un dataloader
    """
    transform = transforms.Compose([
        transforms.Resize(32), # Solve a error for mismatch sizes
        transforms.ToTensor()
    ])
    #return dataManipulator.Lens2(const.PATH_OF_PREDOs_PC, transform)
    return dataManipulator.Lens(const.PATH_OF_FIGURES, 'test', transform)
    
def GAN():
    dataloader = DataLoader(read_data(), const.BATCH_SIZE_128,
                            shuffle=True, num_workers=4)
    generator = FastGAN.Generator()
    discriminator = FastGAN.Discriminator()
    trainer.train(generator, discriminator, dataloader)


def get_cifar_dataloaders(all_transforms):
    # Create Training dataset
    train_dataset = torchvision.datasets.CIFAR10(root = const.PATH_TO_SAVE_MODEL,
                                                train = True,
                                                transform = all_transforms,
                                                download = True)

    # Create Testing dataset
    test_dataset = torchvision.datasets.CIFAR10(root = const.PATH_TO_SAVE_MODEL,
                                                train = False,
                                                transform = all_transforms,
                                                download=True)

    # Instantiate loader objects to facilitate processing
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = const.BATCH_SIZE_64,
                                            shuffle = True)


    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = const.BATCH_SIZE_64,
                                            shuffle = True)
    
    return train_loader, test_loader

def CNN():
    # DataLoading
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    train_dl, test_dl = get_cifar_dataloaders(transform) 

    tools.plot_dataloader(
        train_dl,
        const.PATH_TO_SAVE_FIG,
        '/cifar10.png'
    )

def main():
    CNN()

if __name__ == '__main__':
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
    main()