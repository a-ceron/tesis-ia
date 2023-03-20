"""
###########################################
Esta es una implementación de un modelo WGAN

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
import torch
import torch.nn as nn

import const

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv
        )
