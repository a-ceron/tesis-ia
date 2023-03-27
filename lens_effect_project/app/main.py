from torchvision import transforms

from model.data.dataLoaders import DataLoaderFactory, DataLoaderLabels
from model.utils import const, tools
from model import trainers

def train_and_test_cifar10():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_dataloader, test_dataloader = DataLoaderFactory.get_cifar10(
        const.PATH_TO_SAVE_MODEL,
        True,
        const.BATCH_SIZE_32,
        transform
    )
    classes = DataLoaderLabels.cifar10

    device = tools.select_device()

    trainer = trainers.CNNTrainer(
        classes, 
        train_dataloader,
        device
    )
    state_model = trainer.train()
    trainer.test(test_dataloader)

    torch.save(
        state_model,
        const.PATH_TO_SAVE_MODEL + 'cifar10_model.pth'
    )


def train_and_test_simple_gan():
    # transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])
    # train_dataloader, test_dataloader = DataLoaderFactory.get_cifar10(
    #     const.PATH_TO_SAVE_MODEL,
    #     True,
    #     const.BATCH_SIZE_32,
    #     transform
    # )
    device = tools.select_device()


    trainer = trainers.SimpleGANTrainer(None, device)
    print(trainer)

def main():
    train_and_test_simple_gan()

if __name__ == '__main__':
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
    main()