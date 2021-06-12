import keras.losses
import sklearn.neighbors as skln
import time

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

class KNNModel:
    def __init__(self):
        self.model = None
        self.accuracy = 0
        self.trainTime = 0
        self.testTime = 0

    def learn_knn_model(self, X, Y, neighbours_count = 5):
        start = time.time()
        x = X.reshape(-1, 784)
        knn = skln.KNeighborsClassifier(n_neighbors=neighbours_count)
        self.model = knn.fit(x, Y)
        self.trainTime = time.time() - start

    def test_knn_model(self, X, Y):
        start = time.time()
        x = X.reshape(-1, 784)
        accuracy = self.model.score(x, Y)
        self.accuracy = accuracy
        self.testTime = time.time() - start
        return self.accuracy

class NeuralNetworkModel:

    def __init__(self):
        self.model = None
        self.accuracy = 0

    def learn_neural_network(self, X, Y, X_val, Y_val):
        self.model = Sequential()
        x = X.reshape(-1, 28, 28, 1)
        x_val = X_val.reshape(-1, 28, 28, 1)
        # hidden layer
        self.model.add(Conv2D(50, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        # flatten output of conv
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        # output layer
        self.model.add(Dense(10))
        # looking at the model summary
        self.model.summary()
        # compiling the sequential model
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], optimizer='adam')
        # training the model for 10 epochs
        self.model.fit(x, Y, epochs=10, validation_data=(x_val, Y_val), callbacks=tf.keras.callbacks.ReduceLROnPlateau(patience=2))






