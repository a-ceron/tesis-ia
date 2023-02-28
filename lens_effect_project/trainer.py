from torch import nn
import torch
import time
from model.utils.tools import plot_batch
from torch import optim


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def deprocess(img):
    return img * 127.5 + 127.5

def pipline(real_img, device, generator, discriminator, loss_fn, g_optimizer, d_optimizer, noise_dim):
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
    g_optimizer.step()

    # Train the discriminator
    discriminator.zero_grad()
    real_out = discriminator(real_img)
    disc_real_loss = loss_fn(real_out, real_labels)
    fake_out = discriminator(gen_img.detach())
    disc_fake_loss = loss_fn(fake_out, fake_labels)

    d_loss = (disc_real_loss + disc_fake_loss) / 2.0
    d_loss.backward()
    d_optimizer.step()

    return g_loss, d_loss

def train(generator, discriminator, data_loader, noise_dim=64):
    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_lr = 0.0001
    d_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.9

    generator.to(device)
    generator.apply(weights_init)
    discriminator.to(device)
    discriminator.apply(weights_init)
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))

    loss_fn = nn.BCELoss()
    
    epochs = 500
    plot_inter = 50

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_loss_avg = 0.0
        d_loss_avg = 0.0
        start = time.time()
        for real_img, _ in data_loader:
            g_loss, d_loss = pipline(real_img, 
                                device, 
                                generator, 
                                discriminator, 
                                loss_fn, 
                                g_optimizer, 
                                d_optimizer,
                                noise_dim
                                )
            
            # Metrics
            g_loss_avg += g_loss.item()
            d_loss_avg += d_loss.item()

        g_loss_avg /= len(data_loader)
        d_loss_avg /= len(data_loader)

        if epoch % plot_inter == 0:
            torch.save(generator.state_dict(), f'generator_w_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_w_{epoch}.pth')
            print(f'Time for epoch {epoch} is {time.time()-start:.4f} sec G loss: {g_loss_avg:.4f} D loss: {d_loss_avg:.4f}')
            plot_batch(generator, device, 128, 64)