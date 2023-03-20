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
    device = "cuda:1" if torch.cuda.is_avaliable() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_dim = 256 #Flat para una imagen de 28x28x1
    batch_size = 32
    num_epoch = 50

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)

    transform  = transforms.Compose()
