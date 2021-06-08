# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
import time

def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    # startTimer("hamming_distance")
    X = X.toarray()
    X_train = X_train.toarray()
    # endTimer()
    return (1-X)@X_train.transpose() + X@(1-X_train).transpose()

    # hamming_distance = np.zeros((X.shape[0], X_train.shape[0]))
    # X = X.toarray()
    # X_train = X_train.toarray()
    # for i in range(X.shape[0]):
    #     for j in range(X_train.shape[0]):
    #         distance = 0
    #         for k in range(X.shape[1]):
    #             if X[i, k] != X_train[j, k]:
    #                 distance += 1
    #         hamming_distance[i, j] = distance
    # endTimer()
    # return hamming_distance

    # hamming_distances = []
    # for x in X:
    #     distances = []
    #     for x_train in X_train:
    #         distance = 0
    #         ran = x.shape[1]
    #         for i in range(ran):
    #             if x[0, i] != x_train[0, i]:
    #                 distance += 1
    #         distances.append(distance)
    #     hamming_distances.append(distances)
    # return np.array(hamming_distances)



def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    # startTimer("sort_train_labels_knn")
    # labelMatrix = []
    # for distances in Dist:
    #     indexes = np.argsort(distances, kind='mergesort')
    #     labelMatrix.append(np.take_along_axis(y, indexes, axis=-1))
    # # endTimer()
    # return np.array(labelMatrix)
    return y[np.argsort(Dist, kind='mergesort')]

def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    # startTimer("p_y_x_knn")
    m = np.amax(y)
    matrix = np.zeros((y.shape[0], m+1))
    for rowIndex in range(y.shape[0]):
        for label in range(m+1):
            # sum = 0
            # for i in range(k):
            #     if label == y[rowIndex, i]:
            #         sum += 1
            sum = np.count_nonzero(y[rowIndex, :k] == label)
            matrix[rowIndex, label] = sum/k
    # endTimer()
    return matrix
    # result = []
    # for label in range(np.max(y) + 1):
    #     result.append(np.sum(label == y[:, :k], axis=-1) / k)
    # return np.array(result).transpose()


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    # startTimer("classification_error")
    errors = 0
    for rowIndex in range(p_y_x.shape[0]):
        max = np.argwhere(p_y_x[rowIndex] == np.max(p_y_x[rowIndex]))
        if y_true[rowIndex] != max[max.shape[0]-1, 0]:
            errors += 1
    # endTimer()
    return errors/p_y_x.shape[0]


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    # startTimer("model_selection_knn")
    Dist = hamming_distance(X_val, X_train)
    labels = sort_train_labels_knn(Dist, y_train)
    best_model = [1, 0, []]
    for k in k_values:
        p_y_x = p_y_x_knn(labels, k)
        error = classification_error(p_y_x, y_val)
        if best_model[0] > error:
            best_model[0] = error
            best_model[1] = k
        best_model[2].append(error)
    # endTimer()
    return best_model[0], best_model[1], best_model[2]


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    # startTimer("estimate_a_priori_nb")
    max = np.max(y_train)
    prob = np.empty((4,))
    for i in range(max+1):
        sum = 0
        for y in y_train:
            if y == i:
                sum += 1
        prob[i] = sum/y_train.shape[0]
    # endTimer()
    return prob


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    # startTimer("estimate_p_x_y_nb")
    matrix = np.ones((np.amax(y_train) + 1, (X_train.shape[1])))
    real_X_train = X_train.toarray()

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            x_train_col = real_X_train[:, j]
            nominator = np.count_nonzero(np.logical_and(y_train == i, x_train_col)) + a - 1
            denominator = np.count_nonzero(y_train == i) + a + b - 2
            matrix[i][j] = nominator / denominator
    
    # X_train = X_train.toarray()
    # max = np.max(y_train)
    # matrix = np.zeros((max + 1, X_train.shape[1]))
    # for label in range(max + 1):
    #     for d in range(X_train.shape[1]):
    #         sum = 0
    #         sum2 = 0
    #         for n in range(X_train.shape[0]):
    #             x = X_train[n, d]
    #             if y_train[n] == label:
    #                 sum2 += 1
    #                 if x:
    #                     sum += 1
    #         matrix[label, d] = (sum + a - 1)/(sum2 + a + b - 2)
    # endTimer()
    return matrix


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    # startTimer("p_y_x_nb")
    matrix = np.zeros((X.shape[0], p_x_1_y.shape[0]))
    X = X.toarray()
    for i in range(X.shape[0]):
        nomin = []
        for m in range(p_y.shape[0]):
            sum = 1
            for j in range(X.shape[1]):
                if X[i, j]:
                    sum *= p_x_1_y[m, j]
                else:
                    sum *= (1 - p_x_1_y[m, j])
            sum *= p_y[m]
            nomin.append(sum)
        denomin = np.sum(nomin)
        for m in range(p_y.shape[0]):
            matrix[i, m] = nomin[m]/denomin

    # matrix = np.zeros((X.shape[0], p_x_1_y.shape[0]))
    # X = X.toarray()
    # for i in range(X.shape[0]):
    #     for m in range(p_y.shape[0]):
    #         sum_mian = 0
    #         for m_temp in range(p_y.shape[0]):
    #             sum = 1
    #             for j in range(X.shape[1]):
    #                 if X[i, j]:
    #                     sum *= p_x_1_y[m_temp, j]
    #                 else:
    #                     sum *= (1 - p_x_1_y[m_temp, j])
    #             sum *= p_y[m_temp]
    #             sum_mian += sum
    #         sum_licz = 1
    #         for j in range(X.shape[1]):
    #             if X[i, j]:
    #                 sum_licz *= p_x_1_y[m, j]
    #             else:
    #                 sum_licz *= (1 - p_x_1_y[m, j])
    #         sum_licz *= p_y[m]
    #         matrix[i, m] = sum_licz/sum_mian
    # endTimer()
    return matrix


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    # startTimer("model_selection_nb")
    a_priori = estimate_a_priori_nb(y_train)
    best_error = 1
    best_a = 0
    best_b = 0
    errors = []
    for a in a_values:
        errors.append([])
        for b in b_values:
            p_x_y = estimate_p_x_y_nb(X_train, y_train, a, b)
            p_y_x = p_y_x_nb(a_priori, p_x_y, X_val)
            error = classification_error(p_y_x, y_val)
            errors[len(errors)-1].append(error)
            if best_error > error:
                best_error = error
                best_a = a
                best_b = b
    # endTimer()
    return best_error, best_a, best_b, errors


startTime = (0, "")


def startTimer(funName: str):
    global startTime
    startTime = time.perf_counter(), funName


def endTimer():
    global startTime
    print("Czas wykonania ", startTime[1], ":", time.perf_counter() - startTime[0])
