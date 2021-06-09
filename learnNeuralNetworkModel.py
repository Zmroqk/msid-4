import random

import models
import numpy as np
import os
import data
import pandas as pd

if __name__ == '__main__':
    directoryPath = "/mnt/g/GaborData"
    x_train, y_train = data.load_mnist("fashion-data")
    x_test, y_test = data.load_mnist("fashion-data", kind="t10k")
    nnm = models.NeuralNetworkModel()
    nnm.learn_neural_network(x_train.reshape(-1, 28, 28, 1)/255., y_train, x_test.reshape(-1, 28, 28, 1)/255., y_test)