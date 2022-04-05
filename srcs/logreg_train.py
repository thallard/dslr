import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _plot_cost(costh):  # This function plot the Cost function value
    print('coucou =', costh)
    for cost, c in costh:
        # print(cost)
        plt.plot(range(len(cost)), cost, 'r')
        plt.title("Convergence Graph of Cost Function of type-" + str(c) + " vs All")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.savefig('../plots/' + c + '_cost')
        plt.clf()


def coef_determination(y, predictions):
    for prediction, c in predictions:
        u = ((y - prediction) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()


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
        epsilon = 1e-5
        class1 = -y * np.log(h + epsilon)

        class2 = (1 - y) * np.log(1 - h + epsilon)
        cost = class1 - class2
        return np.sum(cost) / m

    # Gradient descent
    def gradient_descent(self, X, y, h, m):
        gradient_value = np.dot(X.T, (h - y)) / m
        return self.learning_rate * gradient_value

    # Main function with loop gradient descent
    def logistic_regression(self, X, y):
        X = self.normalize(X)
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        print('y vaut = ', y.shape)
        for i in np.unique(y):

            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1], 1))
            print(theta.shape)
            print(X.shape)
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

        _plot_cost(self.cost_history)
        return self

    def predict(self, X):  # this function calls the max predict function to classify the individul feauter
        X = np.insert(X, 0, 1, axis=1)
        sigmoid_v = np.vectorize(self.sigmoid)
        X_predicted = [max((sigmoid_v(i.dot(theta)), c) for theta, c in self.theta)[1] for i in X]
        return X_predicted

    def score(self, X, y):  # This function compares the predictd label with the actual label to find the model perform
        # sigmoid_v = np.vectorize(self.sigmoid)
        predicted = self.predict(X)
        true = 0
        for i in range(len(predicted)):
            if predicted[i] == y[i]:
                true += 1
        # tmp = np.rint(sigmoid_v(X.dot(self.theta[0])))
        # ok = 0
        # for i in range(len(self.Y)):
        #     if tmp[i] == self.Y[i]:
        #         ok += 1
        return true * 100 / len(predicted)
        # score = sum(self.predict(X) == y) / len(y)
        # return score


if __name__ == '__main__':
    data = pd.read_csv('../datasets/dataset_train.csv')

    data = data[data['Herbology'].notna()]
    data = data[data['Defense Against the Dark Arts'].notna()]

    print(len(data))
    for i in range(0, 15):
        data = data.drop(data['Herbology'].idxmax())
        data = data.drop(data['Herbology'].idxmin())
        data = data.drop(data['Defense Against the Dark Arts'].idxmax())
        data = data.drop(data['Defense Against the Dark Arts'].idxmin())
    print(len(data))

    y_data = data['Hogwarts House'].values
    y_data = y_data.reshape(y_data.shape[0], 1)
    # x_data = np.array(data['Astronomy'])
    # x_data = data[['Astronomy', 'Herbology']]

    data = data.iloc[:, 8:10]

    # data = data[data.notna()]
    # data = data[data['Herbology'].notna()]
    x_data = data.values

    print(data.sort_values(by=['Defense Against the Dark Arts']))

    # q_low = df["col"].quantile(0.01)
    # q_hi  = df["col"].quantile(0.99)

    # df_filtered = df[(df["col"] < q_hi) & (df["col"] > q_low)]
    print(x_data)
    # for i in range(len(x_data)):
    #     if x_data[i][1] == np.nan:
    #         print('fdp')
    logi = LogisticRegression(20, 0.1).logistic_regression(x_data, y_data)

    # _plot_cost(logi.cost_history)
    # plt.show()
    # print(logi.theta)
    print(logi.predict(x_data))
    # y_onevsall = np.where(y_data == 'Gryffindor', 1, 0)
    print(logi.score(x_data, y_data))
    # print(len(logi.theta), logi.theta)
    # print(dataset.head())
    print(x_data)
    print(x_data.shape)
    # history = LogisticRegression.cost_history
    # scatter.draw_scatter(pd.read_csv('../datasets/dataset_train.csv'), 'Herbology', 'Astronomy')
    # score = coef_determination(y_data, x_data.dot(theta_final))
