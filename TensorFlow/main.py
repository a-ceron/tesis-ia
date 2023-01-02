from models import VGG, ResNet
from models import models as m
from models.utils import tools

from tensorflow.keras.datasets import cifar10

def VGG16():
    vgg = VGG.My_VGG16()

    vgg.model()
    vgg.train()    
    
def ResNet50():
    resnet = ResNet.My_ResNet()

    resnet.model()
    resnet.train()
    
def CNN():
    print("Downloading CIFAR10 dataset...")
    # Paso 1. Obtenemos los datos   
    cifar = cifar10

    ## Preprocesamiento
    (X_train, y_train), (X_test, y_test) = cifar.load_data()
    X_train, X_test = tools.max_norm(X_train), tools.max_norm(X_test)
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Metadato
    K = len(set(y_train)) # Número de clases
    image_shape = X_train[0].shape
    layers = [1, 1, 1, 2]
    batch_size = 32
    epochs = 5

    print("Creating model...")
    # Paso 2. Creamos el modelo
    cnn = m.CNN(
        num_classes=K,
        input_shape=image_shape,
        layers=layers,
        batch_size=batch_size,
        epochs=epochs
    )

    print("Training model...")
    # Paso 3. Entrenamos el modelo
    model, r = cnn.train(X_train, y_train, X_test, y_test)

    # Paso 4. Evaluamos el modelo
    print("Train score:", model.evaluate(X_train, y_train))
    print("Test score:", model.evaluate(X_test, y_test))



def main():
   CNN()

if __name__ == '__main__':
    main()