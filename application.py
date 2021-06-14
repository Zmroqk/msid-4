import sklearn.neighbors as skln
import numpy as np
from skimage import color
from skimage import io
import features as f
import pandas as pd
from tensorflow.keras.models import load_model
import input as inp

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
    knn_model = False
    model = None
    kernel = None
    model_network = None
    if inp.ask_y_n("Do you want to use KNN model? (y or n)"):
        path_to_knn_model = input("Path to KNN model: ")
        knn_model = True
        model_data = pd.read_pickle(path_to_knn_model)
        model = model_data["model"]
        kernel = model_data["kernel"]
    else:
        path_to_neural_network_model = input("Path to neural network model: ")
        model_network = load_model(path_to_neural_network_model)
    return knn_model, model_network, model, kernel


def run_predict_knn(model_knn: skln.KNeighborsClassifier, model_kernel):
    try:
        path_to_image = input("Path to image: ")
        image = color.rgb2gray(io.imread(path_to_image))
        image = image * 255
        image = np.uint8(image).reshape(28, 28)
        if model_kernel is not None:
            image = f.ApplyGaborSingle(image, model_kernel)
        image = image.reshape(1, -1)
        y = model_knn.predict(image)
        print(f"KNN: {labels[y[0]]}")
    except:
        pass

def run_predict_neural(model_network):
    try:
        path_to_image = input("Path to image: ")
        image = color.rgb2gray(io.imread(path_to_image))
        try:
            image = image.reshape(1, 28, 28, 1)
            y = model_network.predict(image)
        except:
            try:
                image = image.reshape(1, 784)
                y = model_network.predict(image)
            except:
                raise Exception()
        print(f"Neural network: {labels[np.argmax(y)]}")
    except:
        pass