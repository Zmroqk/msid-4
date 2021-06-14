import random
import sys

import models
import numpy as np
import os
import data
import pandas as pd
import input as inp
def __learnKNN_input_data():
    k_values = inp.get_positive_values("Provide k value: ")
    path = input("Path to filtered images: ")
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_learn_log = input("Path where to create log: ")
    path_model = input("Path where to create model: ")
    model_name = input("Name for model: ")
    return k_values, path, path_train_data, path_test_data, path_learn_log, path_model, model_name


def __learnKNN_without_filters_input_data():
    k_values = inp.get_positive_values("Provide k value: ")
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_model = input("Path where to create model: ")
    model_name = input("Name for model: ")
    return k_values, path_train_data, path_test_data, path_model, model_name

def learnKNN():
    k_values, directory_path, path_train_data, path_test_data, path_learn_log, path_model, model_name = __learnKNN_input_data()
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
    pd.to_pickle({'model': best_model, 'kernel': best_kernel, 'label': best_label}, os.path.join(path_model, model_name))


def learnKNN_without_filter():
    k_values, path_train_data, path_test_data, path_model, model_name = __learnKNN_without_filters_input_data()
    x_train, y_train = data.load_mnist(path_train_data)
    x_test, y_test = data.load_mnist(path_test_data, kind="t10k")
    best_model = None
    best_accuracy = 0
    best_label = ""
    for k in k_values:
        knn_model = models.KNNModel()
        knn_model.learn_knn_model(x_train, y_train, k)
        accuracy = knn_model.test_knn_model(x_test, y_test)
        label = f"k value: {k} |\t| accuracy: {knn_model.accuracy} |\t|" \
                f" train time: {knn_model.trainTime} |\t| test time: {knn_model.testTime}"
        if best_accuracy < accuracy:
            best_model = knn_model.model
            best_accuracy = accuracy
            best_label = label
    print(f"Best model: \naccuracy: {best_accuracy}\nlabel: {best_label}")
    pd.to_pickle({'model': best_model, 'kernel': None, 'label': best_label}, os.path.join(path_model, model_name))


def __learn_neural_network_input_data():
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_model = input("Path where to create model: ")
    model_name = input("Name for model: ")
    return path_train_data, path_test_data, path_model, model_name


def learn_neural_network(option: int):
    path_train_data, path_test_data, path_model, model_name = __learn_neural_network_input_data()
    x_train, y_train = data.load_mnist(path_train_data)
    x_test, y_test = data.load_mnist(path_test_data, kind="t10k")
    nnm = models.NeuralNetworkModel()
    if option == 1:
        nnm.learn_neural_network(x_train / 255., y_train, x_test / 255., y_test)
    elif option == 2:
        nnm.learn_neural_network_2(x_train / 255., y_train, x_test / 255., y_test)
    else:
        nnm.learn_neural_network_3(x_train / 255., y_train, x_test / 255., y_test)
    nnm.model.save(os.path.join(path_model, model_name))
