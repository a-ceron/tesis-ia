import torchvision
from torchvision import transforms
from model.GANs import FastGAN
from model.CNNs import CNN, aCNN
from model.data import dataManipulator
from model.utils import const, tools
#import trainer as trainer
from torch import nn
from model import trainers

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

def get_cifar_dataloaders(all_transforms):
    # Create Training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root = const.PATH_TO_SAVE_MODEL,
        train = True,
        transform = all_transforms,
        #download = True
    )

    # Create Testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root = const.PATH_TO_SAVE_MODEL,
        train = False,
        transform = all_transforms,
        #download=True
    )

    # Instantiate loader objects to facilitate processing
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = const.BATCH_SIZE_64,
        shuffle = True
    )


    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = const.BATCH_SIZE_64,
        shuffle = True
    )
    
    return train_loader, test_loader

def convolutional():
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
    device = tools.select_device()

    # Model
    num_classes = 10
    #cnn = ConvNeuralNet(num_classes).to(device)
    cnn = aCNN.ariCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    oprimizer = torch.optim.Adam(
        cnn.parameters(),
        lr=0.001
    )
    # Training
    num_epochs = 20
    device = tools.select_device()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = cnn(images).to(device)
            loss = criterion(outputs, labels).to(device)

            # Backward and optimize
            oprimizer.zero_grad()
            loss.backward()
            oprimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
    # Test
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def cnn_ariel():
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
    num_classes = 10

    trainer = trainers.CNNTrainer(num_classes, train_dl)
    trainer.train()


def main():
    convolutional()

if __name__ == '__main__':
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
    main()