from __future__ import print_function, division
from builtins import range, input

from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from tensorflow.keras.utils import load_img, img_to_array

import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b


tf.compat.v1.disable_eager_execution()

# Función para crear el modelo de VGG16 con AveragePooling
def VGG16_AvgPool(shape):
    # Estamos usando el modelo de VGG16 con pesos de imagenet
    # y vamos a eliminar la última capa (fully-connected)
    # pero vamos a reemplazar el maxpooling por un averagepooling
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    
    new_model = tf.keras.Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())   # Aquí estamos reemplazando el maxpooling por un averagepooling
        else:
            new_model.add(layer)
    return new_model, vgg.layers[0]

# Función para crear el modelo de VGG16 con AveragePooling y con un número de convoluciones
def VGG16_AvgPool_CutOff(shape, num_convs):
    if num_convs < 1 or num_convs > 13:
        # Condición de existencia en las convoluciones
        print("Numero de convoluciones invalido")
        return None
    
    model, init = VGG16_AvgPool(shape)    # Usamos el modelo que creamos
    new_model = tf.keras.Sequential()


    # Creamos un nuevo modelo con las primeras n convoluciones
    n = 0
    new_model.add(init)
    for layer in model.layers:
        new_model.add(layer)
        if layer.__class__ == Conv2D:
            n += 1
        if num_convs >= n:
            break
    return new_model

# Función para preprocesar la imagen
def unpreprocess(img):
    # Resta la media de imagen de imagenet
    img[...,0] += 103.939
    img[...,1] += 116.779
    img[...,2] += 123.68
    # 'BGR' -> 'RGB'
    img = img[...,::-1]
    return img

def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

if __name__ == '__main__':
    # Podemos modificr esta variable
    path = '/Users/aceron/Library/CloudStorage/GoogleDrive-arielcerong@gmail.com/Mi unidad/Aprendizaje/Tesis/Maestría/Code/master-degree/TensorFlow/data/myDataSet/RandomImages/tucan.jpeg'
    img = load_img(path)

    # Preprocesamos la imagen
    x = img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    x = preprocess_input(x)

    batch_shape = x.shape
    shape = x.shape[1:]
    print("batch_shape:", batch_shape)
    print("shape:", shape)

    # Creamos el modelo
    num_convs = 10       # Podemos modificar para ver otros resultados
    model_vgg = VGG16_AvgPool_CutOff(shape, num_convs)

    # Creamos los tensores de entrada y salida
    target = K.variable(model_vgg.predict(x))
    y_hat = model_vgg.output

    # Función de perdida, cuadrado de la diferencia entre la imagen de entrada y la imagen de salida
    loss = K.mean(K.square(target - y_hat))
    grads = K.gradients(loss, model_vgg.input) # gradientes que se quieren optimizar 
    get_loss_and_grads = K.function(
        inputs=[model_vgg.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    from datetime import datetime
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        print("iteracion:", i, "loss:", l)
        losses.append(l)

    print("Tiempo total:", (datetime.now() - t0))
    plt.plot(losses)
    plt.show()

    new_img = x.reshape(*batch_shape)
    final_img = unpreprocess(new_img[0])

    plt.imshow(scale_img(final_img))
    plt.show() 