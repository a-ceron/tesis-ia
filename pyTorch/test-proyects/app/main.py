import const
import models.myFastGAN as FGAN

from models.tools.myDataset import Dataset_Lens

from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import numpy as np

from torch import nn
import torch
from torch import optim
import matplotlib.pyplot as plt
import time


def train(generator, discriminator, g_opt, d_opt, data_loader, 
          loss_fn, epochs, plot_inter, device, noise_dim):
    
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_loss_avg = 0.0
        d_loss_avg = 0.0
        start = time.time()
        for real_img, _ in data_loader:
             # Train the generator
            real_img = real_img.to(device)
            real_img = 2.0 * real_img - 1.0
            
            generator.zero_grad()
            batch_size = real_img.shape[0]
            noise = torch.randn(batch_size, noise_dim, device=device)
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)
            gen_img = generator(noise)
            fake_out = discriminator(gen_img)

            g_loss = loss_fn(fake_out, real_labels)
            g_loss.backward()
            g_opt.step()

            # Train the discriminator
            discriminator.zero_grad()
            real_out = discriminator(real_img)
            disc_real_loss = loss_fn(real_out, real_labels)
            fake_out = discriminator(gen_img.detach())
            disc_fake_loss = loss_fn(fake_out, fake_labels)

            d_loss = (disc_real_loss + disc_fake_loss) / 2.0
            d_loss.backward()
            d_opt.step()
            
            # Metrics
            g_loss_avg += g_loss.item()
            d_loss_avg += d_loss.item()

        g_loss_avg /= len(data_loader)
        d_loss_avg /= len(data_loader)

        if epoch % plot_inter == 0:
            print(f'Time for epoch {epoch} is {time.time()-start:.4f} sec G loss: {g_loss_avg:.4f} D loss: {d_loss_avg:.4f}')
            plot_batch(generator, device)

def deprocess(img):
    return img * 127.5 + 127.5

def plot_batch(generator, device, batch_size, noise_dim):
    name = f'name_{np.random.randint(1000)}'
    n_images = 64
    generator.eval()
    noise = torch.randn(batch_size, noise_dim, device=device)

    gen_batch = generator(noise).detach().cpu()
    plt.figure(figsize=(16, 4))
    
    plot_batch = deprocess(gen_batch)
    for i in range(n_images):
        plt.subplot(4, 16, i+1)
        plt.imshow(np.transpose(plot_batch[i], [1, 2, 0]).numpy().astype("uint8"))
        plt.axis('off')
    plt.savefig(f'{name}')

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

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
    if label == 'test': # Cambiar
        transformer = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(),
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(np.pi),
                # transforms.RandomInvert(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomEqualize()
            ]),
            transforms.Resize(32),
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
    seed = 42
    torch.manual_seed(seed)

    # Data
    label = 'test'
    fig_size = 128
    noise_dim = 64

    batch_size = const.BATCH_SIZE_128
    num_workers = const.TWO_WORKERS
    pin_memory = const.TRUE_VALUE
    shuffle = const.TRUE_VALUE

    test_dataloader = lens_dataloader(label, batch_size, shuffle, num_workers, pin_memory)

    # Model generator
    noise = torch.randn(batch_size, noise_dim)
    test_model_generator = FGAN.Generator(fig_size, noise_dim)
    print(test_model_generator)
    
    test_batch, _ = next(iter(test_dataloader))
    print(test_batch.shape)
    gen_batch = test_model_generator(noise)
    # ([batch, clases, altura, anchura])
    print(gen_batch.shape)  # torch.Size([32, 3, 32, 32])
    

    # Model discriminator
    test_model_discriminator = FGAN.Discriminator(fig_size)
    print(test_model_discriminator)

    dis_batch = test_model_discriminator(test_batch)
    print(dis_batch.shape)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.9

    test_model_generator.to(device)
    test_model_generator.apply(weights_init)
    test_model_discriminator.to(device)
    test_model_discriminator.apply(weights_init)
    g_optimizer = optim.Adam(test_model_generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(test_model_discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))

    loss_fn = nn.BCELoss()

    plot_batch(test_model_generator, device, batch_size, noise_dim)
    
    epochs = 500
    plot_inter = 50

    train(test_model_generator, test_model_generator, g_optimizer, d_optimizer, 
        test_dataloader, loss_fn, epochs, plot_inter, device, noise_dim)
if __name__ == '__main__':
    main()