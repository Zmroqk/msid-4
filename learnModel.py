import random
import sys

import models
import numpy as np
import os
import data
import pandas as pd
import input as inp
def __LearnKNNInputData():
    k_values = inp.get_positive_values("Provide k value: ")
    path = input("Path to filtered images: ")
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_learn_log = input("Path where to create log: ")
    path_model = input("Path where to create model: ")
    return k_values, path, path_train_data, path_test_data, path_learn_log, path_model

def LearnKNN():
    k_values, directory_path, path_train_data, path_test_data, path_learn_log, path_model = __LearnKNNInputData()
    log = open(os.path.join(path_learn_log, "learnKNNlogs.txt"), mode="w+")
    x_train, y_train = data.load_mnist(path_train_data)
    x_test, y_test = data.load_mnist(path_test_data, kind="t10k")
    best_model = None
    best_accuracy = 0
    best_kernel = None
    best_label = ""
    for index in range(len(os.listdir(directory_path))//2):
        x_train_filtered = np.load(os.path.join(directory_path, f"data-train{index}.npz"))['data']
        kernel = np.load(os.path.join(directory_path, f"data-train{index}.npz"))['kernel']
        x_test_filtered = np.load(os.path.join(directory_path, f"data-test{index}.npz"))['data']
        for k in k_values:
            knn_model = models.KNNModel()
            knn_model.learn_knn_model(x_train_filtered, y_train, k)
            accuracy = knn_model.test_knn_model(x_test_filtered, y_test)
            label = f"Tested: data{index} |\t| k value: {k} |\t| accuracy: {knn_model.accuracy} |\t|" \
                    f" train time: {knn_model.trainTime} |\t| test time: {knn_model.testTime}"
            if best_accuracy < accuracy:
                best_model = knn_model.model
                best_accuracy = accuracy
                best_kernel = kernel
                best_label = label
            print(label)
            log.write(label + "\n")
            log.flush()
    print(f"Best model: \naccuracy: {best_accuracy}\nlabel: {best_label}")
    pd.to_pickle({'model': best_model, 'kernel': best_kernel, 'label': best_label}, os.path.join(path_model,
                                                                                                 "modelKNN.pkl"))


def __LearnNeuralNetworkInputData():
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_model = input("Path where to create model")
    return path_train_data, path_test_data, path_model


def LearnNeuralNetwork():
    path_train_data, path_test_data, path_model = __LearnNeuralNetworkInputData()
    x_train, y_train = data.load_mnist(path_train_data)
    x_test, y_test = data.load_mnist(path_test_data, kind="t10k")
    nnm = models.NeuralNetworkModel()
    nnm.learn_neural_network(x_train / 255., y_train, x_test / 255., y_test)
    nnm.model.save(os.path.join(path_model, "modelNeural.h5"))
