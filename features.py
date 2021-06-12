import multiprocessing

import pandas as pd
import numpy as np
import cv2


def GaborFilter(theta_values, sigma_values, lambda_values, gamma_values, ksize_values, phi_values):
    kernels = []
    for theta in theta_values:
        theta = theta / 4. * np.pi
        for sigma in sigma_values:
            for lamda in lambda_values:
                for gamma in gamma_values:
                    for ksize in ksize_values:
                        for phi in phi_values:
                            gabor_label = f"ksize: {ksize}, sigma: {sigma}, theta: {theta}," \
                                          f" lambda: {lamda}, gamma: {gamma}, phi: {phi}"
                            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi,
                                                        ktype=cv2.CV_32F)
                            kernels.append((kernel, gabor_label))
    return kernels


def ApplyGaborSingle(image, kernel):
    image = image.reshape(28, 28)
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)


def ApplyGabor(x, kernels, path, label = "train"):
    index = 0
    for i, kernel in enumerate(kernels):
        filtered_images = []
        for j, image in enumerate(x):
            image = image.reshape(28, 28)
            filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernel[0]))
        np.savez_compressed(f"{path}/data-{label}{index}.npz", data=np.array(filtered_images),
                            kernel=np.array(kernel[0]))
        index += 1


def ApplyGaborSpecific(x, kernels, path, indices, label = "train", verbose=False):
    for i in indices:
        filtered_images = []
        for j, image in enumerate(x):
            image = image.reshape(28, 28)
            filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernels[i][0]))
        if verbose:
            print(f"{path}/data-{label}{i}.npz")
        np.savez_compressed(f"{path}/data-{label}{i}.npz", data=np.array(filtered_images),
                            kernel=np.array(kernels[i][0]))


def ApplyGaborMultithread(x, kernels, path, batchSize, label, startIndex = 0, verbose=False):
    for i in range(startIndex, batchSize+startIndex):
        filtered_images = []
        for j, image in enumerate(x):
            image = image.reshape(28, 28)
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
