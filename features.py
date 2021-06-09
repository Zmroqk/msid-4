import multiprocessing

import pandas as pd
import numpy as np
import cv2


def GaborFilter():
    # Generate Gabor features
    kernels = []  # Create empty list to hold all kernels that we will generate in a loop
    for theta in range(1, 8, 2):  # Define number of thetas. Here only 2 theta values 0 and 1/4 . pi
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5, 7):  # Sigma with values of 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                    for ksize in (3, 4):
                        for phi in (0, 0.4, 0.8, 1):
                            gabor_label = f"ksize: {ksize}, sigma: {sigma}, theta: {theta}," \
                                          f" lambda: {lamda}, gamma: {gamma}, phi: {phi}"
                            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi,
                                                        ktype=cv2.CV_32F)
                            kernels.append((kernel, gabor_label))
    return kernels


def ApplyGaborSingle(image, kernel):
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)


def ApplyGabor(x, kernels, path, label = "train"):
    index = 0
    for i, kernel in enumerate(kernels):
        filtered_images = []
        for j, image in enumerate(x):
            filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernel[0]))
        np.savez_compressed(f"{path}/data-{label}{index}.npz", data=np.array(filtered_images),
                            kernel=np.array(kernel[0]))
        index += 1


def ApplyGaborSpecific(x, kernels, path, indices, label = "train", verbose=False):
    for i in indices:
        filtered_images = []
        for j, image in enumerate(x):
            filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernels[i][0]))
        if verbose:
            print(f"{path}/data-{label}{i}.npz")
        np.savez_compressed(f"{path}/data-{label}{i}.npz", data=np.array(filtered_images),
                            kernel=np.array(kernels[i][0]))


def ApplyGaborMultithread(x, kernels, path, batchSize, label, startIndex = 0, verbose=False):
    for i in range(startIndex, batchSize+startIndex):
        filtered_images = []
        for j, image in enumerate(x):
            filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernels[i][0]))
        if verbose:
            print(f"{path}/data-{label}{i}.npz")
        np.savez_compressed(f"{path}/data-{label}{i}.npz", data=np.array(filtered_images),
                            kernel=np.array(kernels[i][0]))


def StartGaborMultithread(number_of_processes: int, kernels, x, path, label="train", verbose=False):
    batch_size = len(kernels)//number_of_processes

    processes = []
    for i in range(number_of_processes):
        process = multiprocessing.Process(target=ApplyGaborMultithread,
                                          args=(x, kernels, path, batch_size, label, i*batch_size, verbose), name=f"Process {i}")
        processes.append(process)
        process.start()
    process = multiprocessing.Process(target=ApplyGaborMultithread,
                                      args=(x, kernels, path, len(kernels)%number_of_processes, label, len(processes) * batch_size, verbose),
                                      name=f"Process {i}")
    processes.append(process)
    process.start()
    for p in processes:
        p.join()
