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
from model.GANs.aWGAN import ariWDiscriminator, ariWGenerator

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
    def save(self): return self
    

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

    def save(self, name):
        torch.save(self.model.state_dict(), const.PATH_TO_SAVE_MODEL + name)
    

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
        y_pred = self.dis(imgs.to(self.device), True).view(-1)

        loss = F.binary_cross_entropy(y_pred, y_true)
        loss.backward()

        return loss.item()

    def train(self, p_disc=None, p_gen=None, num_epochs=10):
        """Se repinte por el número de épocas
        y por cada época se repite por cada batch

        Seleccionamos un conjunto de puntos del espacio
        lantente y los pasamos por el modelo generador

        """
        z_dim = 100
        img_channels = 3

        dis_loss = 0
        gen_loss = 0

        self.gen = ariGenerator(z_dim).to(self.device)
        self.dis = ariDiscriminator(img_channels).to(self.device)
        
        tools.initialize_weights(self.dis, p_disc)
        tools.initialize_weights(self.gen, p_gen)

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
                    -1,  # Etiqueta real
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
                    1,  # Etiqueta falsa
                )
                optim_dis.step()

                dis_loss = dis_real_loss + dis_fake_loss

                # Entrenamos el modelo generador
                self.gen.zero_grad()
                gen_loss = self._propagator(
                    fake_imgs,
                    fake_imgs.shape[0],
                    -1,  # Etiqueta real
                )
                optim_gen.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {dis_loss:.4f}, Loss G: {gen_loss:.4f}")


        tools.plot_batch(
            self.gen,
            self.device,
            real_imgs.shape[0],
            z_dim
        )
    
    def save(self, name):
        torch.save(self.gen.state_dict(), const.PATH_TO_SAVE_MODEL  + '_gen_' + name)
        torch.save(self.dis.state_dict(), const.PATH_TO_SAVE_MODEL  + '_dis_' + name)

    def test(self):
        raise NotImplementedError
        

class WGANTrainer(Trainer):
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
        self._name = self._name + "WGANTrainer"
        self.dataloader = dataloader
        self.device = device
    
    def _critic_train(self, real_imgs, batch_size, z_dim, iterations, optimizer, weigth_clip):
        for _ in range(iterations):
            noise = torch.randn(
                batch_size,
                z_dim,
                1,
                1,
                device=self.device
            )
            fake_imgs = self.gen(noise)
            
            critic_real = self.dis(real_imgs.to(self.device), True).view(-1)
            critic_fake = self.dis(fake_imgs.detach().to(self.device), True).view(-1)

            loss = -torch.mean(critic_real) + torch.mean(critic_fake)

            self.dis.zero_grad()
            loss.backward(retain_graph=True)

            optimizer.step()

            for p in self.dis.parameters():
                p.data.clamp_(-weigth_clip, weigth_clip)

        return fake_imgs

    def train(self, p_disc=None, p_gen=None, num_epochs=10):
        """Se repinte por el número de épocas
        y por cada época se repite por cada batch

        Seleccionamos un conjunto de puntos del espacio
        lantente y los pasamos por el modelo generador

        """
        z_dim = 100
        img_channels = 3
        criterion_iter = 5
        weigth_clip = 0.01

        dis_loss = 0
        gen_loss = 0

        self.gen = ariWGenerator(z_dim).to(self.device)
        self.dis = ariWDiscriminator(img_channels).to(self.device)
        
        tools.initialize_weights(self.dis, p_disc)
        tools.initialize_weights(self.gen, p_gen)

        optim_gen = torch.optim.RMSprop(
            self.gen.parameters(),
            lr=0.0002,
        )
        optim_dis = torch.optim.RMSprop(
            self.dis.parameters(),
            lr=0.0002,
        )
        
        for epoch in range(num_epochs): # Epocas
            for item, real_imgs in enumerate(self.dataloader):   # Batches
                if isinstance(real_imgs, tuple):
                    real_imgs = real_imgs[0]
                    
                # Entrenamos el modelo discriminador
                fake = self._critic_train(
                    real_imgs,
                    real_imgs.shape[0],
                    z_dim,
                    criterion_iter,
                    optim_dis,
                    weigth_clip
                )

                # Entrenamos el modelo generador
                output = self.dis(fake, True).view(-1)
                loss = -torch.mean(output)
                self.gen.zero_grad()
                loss.backward()
                optim_gen.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {dis_loss:.4f}, Loss G: {gen_loss:.4f}")


        tools.plot_batch(
            self.gen,
            self.device,
            z_dim,
            'wgan'
        )
    
    def save(self, name):
        torch.save(self.gen.state_dict(), const.PATH_TO_SAVE_MODEL  + '_gen_' + name)
        torch.save(self.dis.state_dict(), const.PATH_TO_SAVE_MODEL  + '_dis_' + name)

    def test(self):
        raise NotImplementedError
    

class WGANGPTrainer(Trainer):
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
        self._name = self._name + "WGANGPTrainer"
        self.dataloader = dataloader
        self.device = device

    def _gradient_penalty(self, real_imgs, fake_imgs, batch_size):
        epsilon = torch.rand(
            (batch_size, 1, 1, 1),
            device=self.device
        ).repeat(1, 3, 64, 64)
        interpolated = (
            epsilon * real_imgs + ((1 - epsilon) * fake_imgs)
        )
        interpolated.requires_grad = True

        mixed_scores = self.dis(interpolated, True)

        gradient = torch.autograd.grad(
            inputs=interpolated,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gp = torch.mean((gradient_norm - 1) ** 2)
        return gp
    
    def _critic_train(self, real_imgs, batch_size, z_dim, iterations, optimizer):
        for _ in range(iterations):
            noise = torch.randn(
                batch_size,
                z_dim,
                1,
                1,
                device=self.device
            )
            fake_imgs = self.gen(noise)
            
            critic_real = self.dis(real_imgs.to(self.device), True).view(-1)
            critic_fake = self.dis(fake_imgs.detach().to(self.device), True).view(-1)
            gp = self._gradient_penalty(real_imgs, fake_imgs, batch_size)
            loss = (
                -torch.mean(critic_real) + torch.mean(critic_fake) + 10 * gp
            )

            self.dis.zero_grad()
            loss.backward(retain_graph=True)

            optimizer.step()

        return fake_imgs

    def train(self, p_disc=None, p_gen=None, num_epochs=30):
        """Se repinte por el número de épocas
        y por cada época se repite por cada batch

        Seleccionamos un conjunto de puntos del espacio
        lantente y los pasamos por el modelo generador

        """
        z_dim = 100
        img_channels = 3
        criterion_iter = 5

        dis_loss = 0
        gen_loss = 0

        self.gen = ariWGenerator(z_dim).to(self.device)
        self.dis = ariWDiscriminator(img_channels).to(self.device)
        
        tools.initialize_weights(self.dis, p_disc)
        tools.initialize_weights(self.gen, p_gen)

        optim_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=0.0001,
            betas=(0, 0.9)
        )
        optim_dis = torch.optim.Adam(
            self.dis.parameters(),
            lr=0.0001,
            betas=(0, 0.9)
        )
        
        for epoch in range(num_epochs): # Epocas
            for item, real_imgs in enumerate(self.dataloader):   # Batches
                if isinstance(real_imgs, tuple):
                    real_imgs = real_imgs[0]
                    
                # Entrenamos el modelo discriminador
                fake = self._critic_train(
                    real_imgs,
                    real_imgs.shape[0],
                    z_dim,
                    criterion_iter,
                    optim_dis,
                    weigth_clip
                )

                # Entrenamos el modelo generador
                output = self.dis(fake, True).view(-1)
                loss = -torch.mean(output)
                self.gen.zero_grad()
                loss.backward()
                optim_gen.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {dis_loss:.4f}, Loss G: {gen_loss:.4f}")


        tools.plot_batch(
            self.gen,
            self.device,
            z_dim,
            'wgan'
        )
    
    def save(self, name):
        torch.save(self.gen.state_dict(), const.PATH_TO_SAVE_MODEL  + '_gen_' + name)
        torch.save(self.dis.state_dict(), const.PATH_TO_SAVE_MODEL  + '_dis_' + name)

    def test(self):
        raise NotImplementedError
        