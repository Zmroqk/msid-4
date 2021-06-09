
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

if __name__ == '__main__':
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
    model_data = pd.read_pickle("model.pkl")
    model: skln.KNeighborsClassifier = model_data["model"]
    image = color.rgb2gray(io.imread("TestImage4.png"))

    modelNetwork = load_model("modelNeural.h5")
    image = image.reshape(1, 28, 28, 1)
    y = modelNetwork.predict(image)
    print(f"Neural network: {labels[np.argmax(y)]}")

    image = image * 255
    image = np.uint8(image).reshape(28, 28)
    image = f.ApplyGaborSingle(image, model_data["kernel"])
    image = image.reshape(28, 28)
    image = image.reshape(1, -1)
    y = model.predict(image)
    print(f"KNN: {labels[y[0]]}")


    pass