import pandas as pd
import numpy as np
import sys
from logreg_train import sigmoid


def get_weights():
    weights = []

    try:
        file = open('../../saves/weights.txt', 'r')

        for line in file.readlines():
            values = line.split(', ')
            values[3] = values[3][:len(values[3]) - 1]
            weights.append((values[:3], values[3]))
    except IOError:
        print('Error during weights file manipulation')
    return weights


def predict(weights, X):
    X = np.insert(X, 0, 1, axis=1)
    sigmoid_v = np.vectorize(sigmoid)
    for w in weights:
        print(X[0])
    X_predicted = [max((sigmoid_v(i.dot(weight)), c) for weight, c in weights)[1] for i in X]
    return X_predicted


if __name__ == '__main__':
    # if len(sys.argv) < 2:

    predict(get_weights())
