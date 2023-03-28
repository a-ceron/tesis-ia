"""
###########################################
Esta es una implementación funciones que
permitan entrenar un modelo

Este es un modelo que sirve para clasificar
imágenes. 

autor: Ariel Cerón González
fecha: Marzo 14, 2023

Bajo la tutoria del Dr. Gibran Fuentes
IIMAS, UNAM
###########################################
"""
from torch import nn

from model.CNNs.aCNN import ariCNN
from model.GANs.aGAN import ariDiscriminator, ariGenerator
from model.utils import tools, const

import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self) -> None:
        self._name = "Trainer: "
        self.model = None
    def __str__(self) -> str:
        return self._name
    def train(self): return self
    def test(self): return self
    def save(self):
        if self.model is not None:
            torch.save(self.model.state_dict(), const.PATH_TO_SAVE_MODEL)
        return False
    

class CNNTrainer(Trainer):
    stride = 1
    padding = 1
    kernel_size = 3

    def __init__(self, classes, dataloader, device) -> None:
        super().__init__()
        self._name = self._name + "CNNTrainer"
        self.dataloader = dataloader
        self.classes = classes
        self.num_classes = len(classes)
        self.device = device

    def train(self, num_epochs=20):
        self.model = ariCNN(self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)   # Necesario?
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        for epoch in range(num_epochs):
            for _, (data, targets) in enumerate(self.dataloader):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                # forward
                scores = self.model(data)
                loss = criterion(scores, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))            
        return self.model.state_dict()

    def test(self, test_loader):
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in test_loader:
                data = data.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        tools.plot_dataloader(
            self.dataloader,
            const.PATH_TO_SAVE_FIG,
            'test_CNN_figure_st.png',
            self.classes
        )
    

class SimpleGANTrainer(Trainer):
    """Entrena un modelo GAN
    
    Intentamos entrenar el modelo Discriminador
    para maximizar la probabilidad de assignar una
    etiqueta correcta a las imágenes reales y falsas. 
    Simultaneamente entrenamos el modelo Generador
    para minimizar la perdida

            log(1 - D(G(z)))
    """
    def __init__(self, dataloader, device) -> None:
        super().__init__()
        self._name = self._name + "SimpleGANTrainer"
        self.dataloader = dataloader
        self.device = device

    def _propagator(self, imgs, batch_size, label):
        y_true = torch.full([batch_size], label, dtype=torch.float, device=self.device)
        y_pred = self.dis(imgs.to(self.device)).view(-1)

        loss = F.binary_cross_entropy(y_pred, y_true)
        loss.backward()

        return loss.item()

    def train(self, num_epochs=1, k=1, transfer=False):
        """Se repinte por el número de épocas
        y por cada época se repite por cada batch

        Seleccionamos un conjunto de puntos del espacio
        lantente y los pasamos por el modelo generador

        """
        if transfer:
            self.gen.apply(tools.weights_init)
            self.dis.apply(tools.weights_init)

        z_dim = 100
        img_channels = 3

        dis_loss = 0
        gen_loss = 0

        self.gen = ariGenerator(z_dim).to(self.device)
        self.dis = ariDiscriminator(img_channels).to(self.device)
        
        optim_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        optim_dis = torch.optim.Adam(
            self.dis.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        for epoch in range(num_epochs): # Epocas
            for item, real_imgs in enumerate(self.dataloader):   # Batches
                if isinstance(real_imgs, tuple):
                    real_imgs = real_imgs[0]
                # Entrenamos el modelo generador
                self.dis.zero_grad()
                dis_real_loss = self._propagator(
                    real_imgs,
                    real_imgs.shape[0],
                    1,  # Etiqueta real
                )

                z = torch.randn(
                    real_imgs.shape[0],
                    z_dim,
                    1,
                    1,
                    device=self.device
                )
                fake_imgs = self.gen(z)
                dis_fake_loss = self._propagator(
                    fake_imgs.detach(),
                    fake_imgs.shape[0],
                    0,  # Etiqueta falsa
                )
                optim_dis.step()

                dis_loss = dis_real_loss + dis_fake_loss

                # Entrenamos el modelo generador
                self.gen.zero_grad()
                gen_loss = self._propagator(
                    fake_imgs,
                    fake_imgs.shape[0],
                    1,  # Etiqueta real
                )
                optim_gen.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {dis_loss:.4f}, Loss G: {gen_loss:.4f}")

        return fake_imgs.detach()

    def test(self, test_loader):
        raise NotImplementedError
        