from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from skimage.transform import resize

from tensorflow.keras.utils import load_img, img_to_array

import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from scipy.optimize import fmin_l_bfgs_b
from style_transfer1 import unpreprocess, scale_img, VGG16_AvgPool
from style_transfer2 import gram_matrix, style_loss, minimize

tf.compat.v1.disable_eager_execution()

def load_img_and_preprocess(path, shape=None):
    img = load_img(path)
    if shape:
        img = img.resize(shape)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

if __name__ == '__main__':

    content_image = load_img_and_preprocess("/Users/aceron/Library/CloudStorage/GoogleDrive-arielcerong@gmail.com/Mi unidad/Aprendizaje/Tesis/Maestría/Code/master-degree/TensorFlow/data/myDataSet/RandomImages/tucan.jpeg")

    h, w = content_image.shape[1:3]
    style_image = load_img_and_preprocess("/Users/aceron/Library/CloudStorage/GoogleDrive-arielcerong@gmail.com/Mi unidad/Aprendizaje/Tesis/Maestría/Code/master-degree/TensorFlow/data/myDataSet/RandomImages/style.jpg", (h, w))

    batch_shape = content_image.shape
    shape = content_image.shape[1:]

    vgg, _ = VGG16_AvgPool(shape)
    # vgg, init = VGG16_AvgPool(shape)

    # content_model = Model(init, vgg.layers[13].get_output_at(0))
    content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
    content_target = K.variable(content_model.predict(content_image))

    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in vgg.layers \
        if layer.name.endswith('conv1')
    ]

    style_model = Model(vgg.input, symbolic_conv_outputs)
    style_layers_outputs = [K.variable(y) for y in style_model.predict(style_image)]
    style_weights = list(range(1,6))

    loss = K.mean(K.square(content_model.output - content_target))

    for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
        # gram_matrix() expects a (H, W, C) as input
        loss += w * style_loss(symbolic[0], actual[0])

    grads = K.gradients(loss, vgg.input)

    get_loss_and_grads = K.function(
        inputs=[vgg.input],
        outputs=[loss] + grads
    )

    def get_loss_and_grads_wrapper(x_vec):
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
    plt.imshow(scale_img(final_img))
    plt.show()