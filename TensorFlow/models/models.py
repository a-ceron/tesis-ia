from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from numpy import power
# Para este modelo vamos a introducir un vector de capas
# Cada elemento hace referencia a un número de filtros
# empezando en 32 y multiplicando por 2 cada capa
class CNN:
    def __init__(self, num_classes:int, input_shape:tuple, layers:list, batch_size:int, epochs:int):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.layers = self._gen_layers(layers)
        self.batch_size = batch_size
        self.epochs = epochs

    def _gen_layers(self, layers:list):
        # Aquí se generan las capas
        l = []
        n = len(layers)
        base = 2
        p = 6

        for i in range(n):
            if layers[i] is not None:
                if isinstance(layers[i], list):
                    for j in layers[i]:
                        for _ in range(j):
                            l.append(power(base, p + i))
                else:
                    for _ in range(layers[i]):
                        l.append(power(base, p + i))
        return l


    def block(self, input):
        x = Conv2D(self.layers[0], (3, 3), activation='relu', padding='same')(input)
        x = MaxPooling2D()(x)
        for layer in self.layers[1:]:
            x = Conv2D(layer, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D()(x)
        return x
    
    def model(self):
        # Aquí se crea el modelo
        initial = Input(shape=self.input_shape)
        x = self.block(initial)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=initial, outputs=x)
        return model
    
    def train(self, X_train, y_train, X_test, y_test, optimizer:str='adam', loss:str='sparse_categorical_crossentropy', metrics=['accuracy']):
        model = self.model()
        # Aquí se entrena el modelo
        model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
        self.r = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=self.epochs,
                    batch_size=self.batch_size)
        return model, self.r
    
    def predict(self, model, X_test):
        # Aquí se predice
        return model.predict(X_test)
        
        
        



    



    
