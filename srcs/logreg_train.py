import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import NaN


def _plot_cost(costh):  # This function plot the Cost function value
    for cost, c in costh:
        # print(cost)
        plt.plot(range(len(cost)), cost, 'r')
        plt.title("Convergence Graph of Cost Function of type-" + str(c) + " vs All")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()


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

    def normalize(self, x):
        # print('print le min =', min(x), ' et le max = ', max(x))
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def model(self, x, theta):
        # les shapes =  (1535, 1)  et x =  (1535, 3)  et theta =  (3, 1)  et le resultat des deux  (1535, 1)
        # print('les shapes dex = ', x.shape , ' et theta = ' , theta.shape, ' et le resultat des deux ', x.dot(theta).shape)
        return x.dot(theta)

    # Sigmoid function
    def sigmoid(self, x):
        # x = self.normalize(x)
        if -x > np.log(np.finfo(type(x)).max):
            return 0.0
            # print('je suis x = ', x)

        value = 1 / (1 + np.exp(-x))
        # print('je suis value = ', value.shape)
        # print(value)
        return value

    # Cost function
    def cost_function(self, y, h):
        m = len(y)
        epsilon = 1e-5
        class1 = -y * np.log(h + epsilon)

        class2 = (1 - y) * np.log(1 - h + epsilon)
        cost = class1 - class2
        return np.sum(cost) / m
        # value = (1 / m) * np.sum(-y * np.log(h)) + (1 - y) * np.log(1 - h)


        # value = (1 / m) * (np.sum(np.log(h) + (1 - y) * np.log(1 - h)))
        # value = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        # return value

    # Gradient descent
    def gradient_descent(self, X, y, h, theta):
        # X = self.normalize(X)
        gradient_value = np.dot(X.T, (h - y)) / len(y)
        theta -= self.learning_rate * gradient_value
        return theta

    # Main function with loop gradient descent
    def logistic_regression(self, X, y):
        X = self.normalize(X)
        X =  np.insert(X, 0, 1, axis=1)
        print('y vaut = ', y.shape)
        for i in np.unique(y):
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1], 1))
            # theta = np.ones(X.shape[1])
            # theta = theta.reshape(theta.shape[0], 1)
            print(theta.shape)
            print(X.shape)
            cost_hist = []
            for j in range(self.iterations):
                # X = self.normalize(X)
                z = self.model(X, theta)
                # print(z)
                # z = self.normalize(z)
                # h = self.sigmoid(z)
                tmp = np.vectorize(self.sigmoid)
                h = tmp(z)
                print(h)

                theta = self.gradient_descent(X, y_onevsall, h, theta)
                cost_hist.append(self.cost_function(y_onevsall, h))
            self.theta.append((theta, i))
            self.cost_history.append((cost_hist, i))
        return self

    def predict(self, X, theta):  # this function calls the max predict function to classify the individul feauter

        X = np.insert(X, 0, 1, axis=1)
        # theta = np.zeros((X.shape[1], 1))
        sigmoid_v = np.vectorize(self.sigmoid)
        # X_predicted = [np.max((sigmoid_v(np.array(i).dot(theta)), c) for theta, c in self.theta)[1] for i in X]
        # sigmoid_v = np.vectorize(self.sigmoid)
        # X_predicted = 0
        # for i in theta:
            # X_predicted -= sigmoid_v(X.dot(theta))
        print(self.theta[0][1])
        return sigmoid_v(X.dot(self.theta[0]))

    def score(self, X, y):  # This function compares the predictd label with the actual label to find the model performance
        score = sum(self.predict(X.dot(np.where())) == y) / len(y)
        return score


data = pd.read_csv('../datasets/dataset_train.csv')

data = data[data['Astronomy'].notna()]
data = data[data['Herbology'].notna()]

y_data = data['Hogwarts House'].values

# x_data = data.drop(['Index', 'Hogwarts House', 'Best Hand', 'First Name', 'Last Name', 'Birthday'], axis=1).values

x_data = data.loc[:, 'Arithmancy':'Herbology'].values
x_data = np.delete(x_data, 0, 1)


y_data = y_data.reshape(y_data.shape[0], 1)
print(x_data.shape)
print(y_data.shape)

print(data.head())
# X_train,X_test,y_train,y_test = train_test_split(X,y_data,test_size = 0.33)
logi = LogisticRegression(10, 0.1).logistic_regression(x_data, y_data)

# print(cost)
# _plot_cost(logi.cost_history)
# plt.show()
# print(logi.theta)
print(logi.predict(x_data, logi.theta))
# print(logi.score(x_data, y_data))
# score = coef_determination(y_data, x_data.dot(theta_final))
