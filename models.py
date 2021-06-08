import sklearn.neighbors as skln
import time

class KNNModel:
    def __init__(self):
        self.model = None
        self.accuracy = 0
        self.trainTime = 0
        self.testTime = 0

    def learn_knn_model(self, X, Y, neighboursCount = 5):
        start = time.time()
        knn = skln.KNeighborsClassifier(n_neighbors=neighboursCount)
        self.model = knn.fit(X, Y)
        self.trainTime = time.time() - start

    def test_knn_model(self, X, Y):
        start = time.time()
        accuracy = self.model.score(X, Y)
        self.accuracy = accuracy
        self.testTime = time.time() - start
        return self.accuracy

