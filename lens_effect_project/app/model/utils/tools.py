import matplotlib.pyplot as plt
import torch
import numpy as np


import torch
import torch.nn as nn

def deprocess(img):
    return img * 127.5 + 127.5

def plot_batch(generator, device, batch_size, noise_dim):
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
    n = np.random.randint(1000)
    plt.savefig(const.PATH_TO_SAVE_FIG+f'/{n}.png')
    
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])


# Vamos a solucionat el ticket
# https://github.com/a-ceron/tesis-ia/issues/12
def select_device(current:int=0):
    if torch.cuda.is_available():
        device = "cuda:{}"
        devices = torch.cuda.device_count()
        print(device.format(devices))
        if devices > 1:
            c_device = torch.cuda.current_device()
            if current == c_device:
                return torch.device(
                    device.format(
                        next_element(
                            c_device, devices
                        )
                    )
                )
        elif devices == 0:
            return torch.cuda.device(devices)
    print('cpu')
    return torch.device('cpu')

def next_element(current:int, m_value:int):
    current += 1
    if current >= m_value:
        return 0
    return current
    


#################
# Parte 2
#################
"""
Vamos a mejorar las herramientas para que puedan ser usadas
en todos los modelos o clases que se creen en el futuro.
"""

def get_figure(labels, images):
    item = torch.randint(
        len(images),
        size=(1,)
    ).item()
    return labels[item], images[item]

# Visualización de datos
def plot_dataloader(data, path, name, n_images=3):
    batch_data, batch_labels = next(iter(data))
    cols, rows = n_images, n_images
    n_items = cols * rows + 1

    figure = plt.figure(figsize=(8, 8))
    for item in range(1, n_items):
        figure.add_subplot(rows, cols, item)
        label, img = get_figure(batch_labels, batch_data)
        plt.title(label)
        plt.imshow(img.permute(1, 2, 0).squeeze())
        plt.axis('off')
    plt.savefig(path + name)
