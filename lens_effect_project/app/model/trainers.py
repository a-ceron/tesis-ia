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
from model.CNNs.aCNN import ariCNN

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

    def __init__(self, num_classes, dataloader) -> None:
        super().__init__()
        self._name = self._name + "CNNTrainer"
        self.num_classes = num_classes
        self.dataloader = dataloader

    def train(self):
        model = ariCNN(self.num_classes)
        data, label = next(iter(self.dataloader))
        output = model(data)
        print(data.shape)
        print(output.shape)

    def test(self):
        pass

    def save(self):
        pass

    def __str__(self):
        return self._name