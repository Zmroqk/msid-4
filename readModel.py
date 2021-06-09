
import sklearn.neighbors as skln
import numpy as np
from skimage import color
from skimage import io
import data as dt
import features as f
import pandas as pd

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
    image = color.rgb2gray(io.imread("TestImage4.png")) * 255
    image = np.uint8(image)
    image = f.ApplyGaborSingle(image, model_data["kernel"])
    image = image.reshape(28, 28)
    image = image.reshape(1, -1)
    y = model.predict(image)
    label = labels[y[0]]
    pass