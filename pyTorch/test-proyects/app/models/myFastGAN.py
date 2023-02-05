import torch
from torch import nn

class FastGAN(nn.Module):
    """ TOWARDS FASTER AND STABILIZED GAN TRAINING FOR HIGH-FIDELITY FEW-SHOT IMAGE SYNTHESIS

    Training Generatice Adversarial Networks (GAN)
    this model try to develop the few-shot image 
    synthesis task for GAN with minimum computing
    cost.

    With this model we expect get a model convergency 
    from scratch in less than a pair of hours of 
    traning on a sigle RTX-2080. [Modelo por definir]

    Also we include a pre-trained model using [cita]
    and data aumentation to generate more examples.
    """
    def __init__(self):
        super(FastGAN, self).__init__()


    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, filters=128, noise_dim=64):
        super(Generator, self).__init__()
        self.filters = filters
        self.init = nn.Sequential(
            nn.Linear(noise_dim, filters * 4 * 4),
            nn.BatchNorm1d(filters * 4 * 4),
            nn.LeakyReLU(0.2),
        )             
        
        self.feat_8   = upBlock(filters, filters)
        self.feat_16  = upBlock(filters, filters // 2)
        self.feat_32  = upBlock(filters // 2, filters // 4)
        
        self.ch_conv = nn.Conv2d(filters // 4, 3, 3, 1, 1, bias=False)

    def forward(self, z):
        feat_4 = self.init(z)
        feat_4 = torch.reshape(feat_4, (-1, self.filters, 4, 4))
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        img = torch.tanh(self.ch_conv(feat_32))
        return img
    
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2)
    )
    return block


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, filters=128, noise_dim=64):
        super(Discriminator, self).__init__()
        self.down_from_big = nn.Sequential(
            nn.Conv2d(3, filters // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(filters // 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.down_16 = DownBlockComp(filters // 8, filters // 4)
        self.down_8 = DownBlockComp(filters // 4, filters // 2)
        self.down_4 = DownBlockComp(filters // 2, filters)

        self.logits = nn.Sequential(
            nn.Conv2d(filters, filters, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.down_from_big(x)
        x = self.down_16(x)
        x = self.down_8(x)
        x = self.down_4(x)
        x = self.logits(x).view(-1, 1)
        return x
    
