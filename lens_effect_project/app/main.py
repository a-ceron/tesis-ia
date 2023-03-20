from torchvision import transforms
from model import dataManipulator, const, FastGAN
import lens_effect_project.app.trainer as trainer

from torch.utils.data import DataLoader

def read_data():
    """Regresa un dataloader
    """
    transform = transforms.Compose([
        transforms.Resize(32), # Solve a error for mismatch sizes
        transforms.ToTensor()
    ])
    #return dataManipulator.Lens2(const.PATH_OF_PREDOs_PC, transform)
    return dataManipulator.Lens(const.PATH_OF_FIGURES, 'test', transform)
    

def main():
    dataloader = DataLoader(read_data(), const.BATCH_SIZE_128,
                            shuffle=True, num_workers=4)
    generator = FastGAN.Generator()
    discriminator = FastGAN.Discriminator()
    trainer.train(generator, discriminator, dataloader)

if __name__ == '__main__':
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
    main()