import sys

import sklearn.neighbors as skln
import numpy as np
from skimage import color
from skimage import io
import data as dt
import features as f
import pandas as pd
import models as md
from tensorflow.keras.models import load_model
import matplotlib.pyplot as pl

labels = [
    "T - shirt / top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle",
    "boot"
]


def init_app():
    path_to_knn_model = input("Path to KNN model: ")
    path_to_neural_network_model = input("Path to neural network model: ")
    model_data = pd.read_pickle(path_to_knn_model)
    model: skln.KNeighborsClassifier = model_data["model"]
    kernel = model_data["kernel"]
    model_network = load_model(path_to_neural_network_model)
    return model_network, model, kernel


def run_predict(model_network, model_knn: skln.KNeighborsClassifier, model_kernel):
    try:
        path_to_image = input("Path to image: ")
        image = color.rgb2gray(io.imread(path_to_image))
        image = image.reshape(1, 28, 28, 1)
        y = model_network.predict(image)
        print(f"Neural network: {labels[np.argmax(y)]}")

        image = image * 255
        image = np.uint8(image).reshape(28, 28)
        image = f.ApplyGaborSingle(image, model_kernel)
        image = image.reshape(1, -1)
        y = model_knn.predict(image)
        print(f"KNN: {labels[y[0]]}")
    except:
        pass