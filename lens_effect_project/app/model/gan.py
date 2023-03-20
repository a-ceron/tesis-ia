"""
###########################################
Esta es una implementación de un modelo GAN

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
import torch
import torch.nn as nn
from torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim

import dataManipulator as dm

import const


### Primera parte
#   Dos redes neuronales, una llamada discriminador
#   la cual se encarga de verificar que una imágen sea
#   o no una imagen real
#
#   Un modelo generador que a partir de un vector z
#   pueda generar una imágen que tenga un gran parecido
#   a las imágenes del conjunto de entrenamiento.
#
#
class Discriminator(nn.Module):
    """ Regresa una probabilidad

    Regresa la probabilidad de que una imágen de entrada
    seo a no real
    """
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(
                in_features=img_dim,
                out_features=128
            ), # Flat
            nn.LeakyReLU(0.1),  # activación para elementos mayores a cero, con una pendiente
            nn.Linear(
                128,
                1
            ),  # Reducimos la dimensión a uno
            nn.Sigmoid()    # Tenemos una probabilidad entre cero y uno, se o no ser
        )

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(
                z_dim,  
                256
            ),  # Aumentado de datos
            nn.LeakyReLU(0.1),  # Activación
            nn.Linear(
                256,
                img_dim
            ), # Aumentado de datos    
            nn.Tanh()   # Genera valores entre -1 y 1
        )

    def forward(self, x):
        return self.model(x)
    

# Vamos a solucionat el ticket
# https://github.com/a-ceron/tesis-ia/issues/12
def select_device(current:int=0):
    if torch.cuda.is_available():
        device = f'cuda:{}'
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
    if current > m_value:
        return 0
    return current
    

 

## Parte 2 
# Tenemos la función de entrenamiento que
# va a poner a competir los dos modelos entre sí
# para lograr que el modelo generador le gane
# al modelo discriminidar}
#
def main():
    ### Metaparámetros

    # Selección de un dispositivo
    device = select_device()
    lr = 3e-4
    z_dim = 64
    image_dim = 784 #Flat para una imagen de 28x28x1
    batch_size = 32
    num_epoch = 50

    disc = Discriminator(image_dim)
    gen = Generator(z_dim, image_dim)

    noise = torch.rand((batch_size, image_dim)) #matriz de 32 x 784
                                                # en este caso veamoslo como
                                                # una lista de vectores.
                                                # 32 vectores de 784 elementos

    transform  = transforms.Compose(
        transform.ToTensor(),
        transform.Normalize((0.5,), (0.5,)),
    )

    dataset = dm.Lens2(const.PATH_OF_PREDOs_PC, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer_disc = optim.Adam(
                disc.parameters(),  # Target to optimai
                lr=lr   # LEARNING RANGE
            )
    optimizer_gen = optim.Adam(
                gen.parameters(),  # Target to optimai
                lr=lr   # LEARNING RANGE
            )
    
    # Loss acá vamos a usar una de las funciones de pérdida
    criterion = nn.BCELoss()

    # Entrenamiento
    step = 0
    for epoch in range(num_epoch):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.view(-1, image_dim) # Acá andamos haciendo un flat a todo
            batch_size = real.shape[0]  ## ???

            ## Train discriminator
            z_noise = torch.rand((batch_size, z_dim))
            fake = gen(z_noise)

            disc_real = disc(real).view(-1) # Flat everithing
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward()
            optimizer_disc.step()

            # generador
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backwards()
            optimizer_gen.step()


            if batch_idx % 50 == 0:
                print(
                    f" Epoca: {epoch/num_epoch}, Loss Generator: {lossG:.4f}, Loss Discriminator: {lossD:.4f}"
                )