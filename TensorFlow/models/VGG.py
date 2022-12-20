from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from models.utils import tools, consts

class My_VGG16:
    def __init__(self):
        self.vgg  = VGG16(input_shape=consts.IMAGE_SIZE_100 + [3], 
                            weights='imagenet', # Pesos pre-entrenados de ImageNet
                            include_top=False # No incluye la capa densa
                            )
        # don't train existing weights
        for layer in self.vgg.layers:
            layer.trainable = False

        self.train_files = consts.DATA_PATH + '/Training'
        self.test_files = consts.DATA_PATH + '/Test'
        

    def model(self):
        folders = tools.get_classes(self.train_files)
        # A partir de aquí puedes agregar capas
        x = Flatten()(self.vgg.output)
        prediction = Dense(len(folders), activation='softmax')(x)

        # create a model object
        self.model = Model(inputs=self.vgg.input, outputs=prediction)

        # view the structure of the model
        print("Model structure:")
        self.model.summary()
    
    def data(self):
        gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input
        )

        # Acá nomas para ver qué pasa
        test_gen = gen.flow_from_directory(self.test_files, target_size=consts.IMAGE_SIZE_100)
        print(test_gen.class_indices)

        labels = [None] * len(test_gen.class_indices)
        for k, v in test_gen.class_indices.items():
            labels[v] = k

        for x, y in test_gen:
            print("min:", x[0].min(), "max:", x[0].max())
            print("shape:", x[0].shape, "label:", labels[np.argmax(y[0])])
            plt.title(labels[np.argmax(y[0])])
            plt.imshow(x[0])
            plt.show()
            break

        # Aca lo bueno
        train_generator = gen.flow_from_directory(self.train_files,
                                                target_size=consts.IMAGE_SIZE_100,
                                                shuffle=True,
                                                batch_size=consts.THITTY_TWO_BATCH_SIZE)
        valid_generator = gen.flow_from_directory(self.test_files,
                                                target_size=consts.IMAGE_SIZE_100,
                                                shuffle=True,
                                                batch_size=consts.THITTY_TWO_BATCH_SIZE)
        return train_generator, valid_generator, gen

    def train(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        
        train_generator, valid_generator, gen = self.data()
        r = self.model.fit_generator(
            train_generator,
            validation_data=valid_generator,   
            epochs=consts.TEN_EPOCHS,
            steps_per_epoch=len(self.train_files) // consts.THITTY_TWO_BATCH_SIZE,
            validation_steps=len(self.test_files) // consts.THITTY_TWO_BATCH_SIZE,
        )

        self.plot(r, train_generator, valid_generator, gen)
        return r
    
    def plot(self, r, train_generator, test_generator, gen):
        cm = tools.get_confusion_matrix(self.train_files, len(self.train_files), gen, consts.IMAGE_SIZE_100, True, self.model, consts.THITTY_TWO_BATCH_SIZE)
        print(cm)
        test_cm = tools.get_confusion_matrix(self.test_files, len(self.test_files), gen, consts.IMAGE_SIZE_100, True, self.model, consts.THITTY_TWO_BATCH_SIZE)
        print(test_cm)

        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        plt.savefig('LossVal_loss')

        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        plt.savefig('AccVal_acc')

