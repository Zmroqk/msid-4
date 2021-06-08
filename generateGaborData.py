import features
from data import load_mnist

if __name__ == '__main__':
    kernels = features.GaborFilter()
    x_train, y_train = load_mnist("fashion-data")
    x_test, y_test = load_mnist("fashion-data", kind="t10k")
    # features.ApplyGabor(x_train, kernels, "/mnt/g/GaborData")
    # features.ApplyGabor(x_test, kernels, "/mnt/g/GaborData", label="test")
    features.StartGaborMultithread(5, kernels, x_train, "/mnt/g/GaborData", verbose=True)
    features.StartGaborMultithread(5, kernels, x_test, "/mnt/g/GaborData", "test", verbose=True)
    # features.ApplyGaborSpecific(x_train, kernels, "/mnt/g/GaborData", [0, 1, 2, 204, 205, 205, 408, 409, 410, 612, 613, 614, 816, 817, 818, 1020, 1021, 1022], verbose=True)
    # features.ApplyGaborSpecific(x_test, kernels, "/mnt/g/GaborData", [0, 1, 2, 3, 1020, 1021, 1022, 1023], label="test")
    pass
