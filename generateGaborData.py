import features
from data import load_mnist
from input import get_positive_values
import numpy as np

def GenerateGaborInputData():
    path_train_data = input("Path to fashion mnist train images: ")
    path_test_data = input("Path to fashion mnist test images: ")
    path_save_location = input("Path where to save data: ")
    theta_values = get_positive_values("Provide theta values: ")
    sigma_values = get_positive_values("Provide sigma values: ")
    lambda_values = np.pi / np.array(get_positive_values("Provide lambda values: "))
    gamma_values = get_positive_values("Provide gamma values: ", use_float=True)
    ksize_values = get_positive_values("Provide ksize values: ")
    phi_values = get_positive_values("Provide phi values: ", use_float=True)
    return path_train_data, path_test_data, path_save_location, theta_values, sigma_values, lambda_values, \
                                                               gamma_values, ksize_values, phi_values


def GenerateGaborData():
    path_train_data, path_test_data, path_save_location, theta_values, sigma_values, lambda_values, \
    gamma_values, ksize_values, phi_values = GenerateGaborInputData()
    kernels = features.GaborFilter(theta_values, sigma_values, lambda_values, gamma_values, ksize_values, phi_values)
    x_train, y_train = load_mnist(path_train_data)
    x_test, y_test = load_mnist(path_test_data, kind="t10k")
    features.StartGaborMultithread(5, kernels, x_train, path_save_location, verbose=True)
    features.StartGaborMultithread(5, kernels, x_test, path_save_location, "test", verbose=True)
    pass
