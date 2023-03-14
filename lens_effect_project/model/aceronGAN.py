"""
###########################################
Esta es una implementación de un modelo GAN propio

autor: Ariel Cerón González
fecha: Marzo 7, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from torch import nn


class Generator(nn.Module):
    """A generator (G) attempts to create a realistic image

    A generative $G$ parametizer by $\theta$ and recives random
    noice $Z$ as input and output will be sample

    $$
    y = G(z,\theta)
    $$

    Moreover, there are a massive training data $x$ recived from
    $p_{data}$ and the objetive of $G$ is to approximate the $p_{data}$
    distribution while using a $p_{g}$ distribution.

    Inputs:
    -------
    z: A random noise vector

    Output:
    -------
    G(z): return synthetic data
    """
    def __init__(self) -> None:
        super(Generator, self).__init__()
    
class Discriminator(nn.Module):
    """A discriminator (D) makes the decision that a random sample either is real o fake


    """
    def __init__(self) -> None:
        super(Discriminator, self).__init__()


# No va a haber mucho cambio con fresener inceptions distnace
# Ver que no hay un colapso de moda al identificar 