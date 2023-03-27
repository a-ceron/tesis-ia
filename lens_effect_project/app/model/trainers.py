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
from model.utils import tools, const

import torch


class Trainer:
    def __init__(self) -> None:
        self._name = "Trainer: "
    def train(self): return self
    def test(self): return self
    def save(self): return self
    def resume(self): return self

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

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return self 

    def __str__(self):
        return self._name