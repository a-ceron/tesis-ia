'''
A residual neural network (ResNet)[1] is an artificial neural network (ANN). It is a gateless or open-gated variant of the HighwayNet,[2] the first working very deep feedforward neural network with hundreds of layers, much deeper than previous neural networks.
'''
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.utils import tools, consts

import matplotlib.pyplot as plt

class My_ResNet:
    def __init__(self) -> None:
        self.resnet = ResNet50(input_shape=consts.IMAGE_SIZE_100 + [3],
                              weights='imagenet',
                              include_top=False)
        for layer in self.resnet.layers:
            layer.trainable = False
        

        self.train_files = consts.DATA_PATH + '/Training'
        self.test_files = consts.DATA_PATH + '/Test'

    def model(self):
        folders = tools.get_classes(self.train_files)
        x = Flatten()(self.resnet.output)
        prediction = Dense(len(folders), activation='softmax')(x)
        self.model = Model(inputs=self.resnet.input, outputs=prediction)
        print("Model structure:")
        self.model.summary()
        self.model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    
    
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

        self.plot(r, gen)
        return r
    
    def plot(self, r, gen):
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

