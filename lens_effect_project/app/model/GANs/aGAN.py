"""
###########################################
Esta es una implementación de una arquitecura
GAN, basado en el artículo Unsupervised 
Representation Learning With Deep Convolutional
Generative Adversarial Networks

Unsupervised Representation Learning With Deep 
Convolutional Generative Adversarial Networks. 
Architecture guidelines for stable Deep Convolutional 
GANs:
- Replace any pooling layers with strided convolutions
(discriminator) and fractional-strided convolutions
(generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the
output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from torch import nn

class ariDiscriminator(nn.Module):
    def __init__(self, channels_img):
        super(ariDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_img,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)


class ariGenerator(nn.Module):
    """Regresa una imagen
    """
    def __init__(self, z_dim):
        super(ariGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=z_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )
        
        

    def forward(self, X):
        return self.model(X)

def test():
    import torch

    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.rand((N, in_channels, H, W))
    z = torch.randn((N, z_dim, 1, 1))

    disc = ariDiscriminator(in_channels)
    out = disc(x)
    assert out.shape == (N, 1, 1, 1)
    print(f"Expected {x.shape}, got {out.shape}")

    gen = ariGenerator(z_dim)
    out = gen(z)
    assert out.shape == (N, in_channels, H, W)
    print(f"Expected {x.shape}, got {out.shape}")

test()