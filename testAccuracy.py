import sys

import sklearn.neighbors as skln
import numpy as np
from data import load_mnist
import features as f
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


def __test_models_input():
    path_model = input("Path to KNN model: ")
    path_neural_model = input("Path to neural network model: ")
    path_test_data = input("Path to fashion mnist test images: ")
    return path_model, path_neural_model, path_test_data


def test_models():
    path_model, path_neural_model, path_test_data = __test_models_input()
    model_data = pd.read_pickle(path_model)
    model: skln.KNeighborsClassifier = model_data["model"]
    modelNetwork = load_model(path_neural_model)
    x_test, y_test = load_mnist(path_test_data, kind="t10k")
    filtered_images = []
    for i, image in enumerate(x_test):
        filtered_images.append(f.ApplyGaborSingle(image, model_data["kernel"]))
    y = model.predict(np.array(filtered_images).reshape(-1, 784))
    print(f"KNN score: {accuracy_score(y, y_test)}")
    y = np.argmax(modelNetwork.predict(x_test.reshape(-1, 28, 28, 1) / 255.), axis=1)
    print(f"Neural network score: {accuracy_score(y, y_test)}")
