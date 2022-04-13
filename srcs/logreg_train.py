import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_cost(costh):  # This function plot the Cost function value
    for cost, c in costh:
        plt.plot(range(len(cost)), cost, 'r')
        plt.title("Convergence Graph of Cost Function of type-" + str(c) + " vs All")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.savefig('../plots/' + c + '_cost')
        plt.clf()


class LogisticRegression(object):
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.cost_history = []
        self.theta = []
        self.time = time.time()

    # Util function which allow to get execution time
    def log(self, msg):
        print("\033[32m" + msg + str(round(time.time() - self.time, 1)) + "s.\033[0;0m")

    def normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def model(self, x, theta):
        return x.dot(theta)

    # Sigmoid function
    def sigmoid(self, x):
        if -x > np.log(np.finfo(type(x)).max):
            return 0.0
        return 1 / (1 + np.exp(-x))

    # Cost function
    def cost_function(self, y, h, m):
        return (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))

    # Gradient descent
    def gradient_descent(self, X, y, h, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        return self.learning_rate * gradient_value

    # Main function with loop gradient descent
    def logistic_regression(self, X, y):
        X = self.normalize(X)
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)

        for i in np.unique(y):
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1], 1))
            cost_hist = []

            for j in range(self.iterations):
                z = self.model(X, theta)
                tmp = np.vectorize(self.sigmoid)
                h = tmp(z)
                theta -= self.gradient_descent(X, y_onevsall, h, m)
                cost_hist.append(self.cost_function(y_onevsall, h, m))

            self.log('Logistic Regression for ' + str(i) + ' class finished in : ')
            self.theta.append((theta, i))
            self.cost_history.append((cost_hist, i))

        plot_cost(self.cost_history)
        return self

    def predict(self, X):  # this function calls the max predict function to classify the individul feauter
        X = np.insert(X, 0, 1, axis=1)
        sigmoid_v = np.vectorize(self.sigmoid)
        X_predicted = [max((sigmoid_v(i.dot(theta)), c) for theta, c in self.theta)[1] for i in X]
        return X_predicted

    def score(self, X, y):  # This function compares the predictd label with the actual label to find the model perform
        predicted = self.predict(X)
        true = 0
        for i in range(len(predicted)):
            if predicted[i] == y[i]:
                true += 1
        return true * 100 / len(predicted)


if __name__ == '__main__':
    data = pd.read_csv('../datasets/dataset_train.csv')

    data = data[data['Herbology'].notna()]
    data = data[data['Defense Against the Dark Arts'].notna()]

    for i in range(0, 20):
        data = data.drop(data['Herbology'].idxmax())
        data = data.drop(data['Herbology'].idxmin())
        data = data.drop(data['Defense Against the Dark Arts'].idxmax())
        data = data.drop(data['Defense Against the Dark Arts'].idxmin())

    y_data = data['Hogwarts House'].values
    y_data = y_data.reshape(y_data.shape[0], 1)

    data = data.iloc[:, 8:10]

    x_data = data.values
    logi = LogisticRegression(3000, 0.05).logistic_regression(x_data, y_data)
    print('Score :', round(logi.score(x_data, y_data), 2))
