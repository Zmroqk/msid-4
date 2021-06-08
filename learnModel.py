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
    best_model = None
    best_accuracy = 0
    best_kernel = None
    for index in range(1024):
        x_train_filtered = np.load(directoryPath + "/" + f"data-train{index}.npz")['data']
        kernel = np.load(directoryPath + "/" + f"data-train{index}.npz")['kernel']
        x_train_filtered = x_train_filtered.reshape((60000, 784))
        x_test_filtered = np.load(directoryPath + "/" + f"data-test{index}.npz")['data']
        x_test_filtered = x_test_filtered.reshape((10000, 784))
        indices = np.random.choice(10000, 4000, replace=False)
        for k in (11, 61):
            knn_model = models.KNNModel()
            knn_model.learn_knn_model(x_train_filtered, y_train, k)
            accuracy = knn_model.test_knn_model(x_test_filtered[indices], y_test[indices])
            if best_accuracy < accuracy:
                best_model = knn_model.model
                best_accuracy = accuracy
                best_kernel = kernel
            print(f"Tested: data{index} |\t| k value: {k} |\t| accuracy: {knn_model.accuracy} |\t|"
                  f" train time: {knn_model.trainTime} |\t| test time: {knn_model.testTime}")
    pd.to_pickle({'model': best_model, 'kernel': best_kernel}, "model.pkl")