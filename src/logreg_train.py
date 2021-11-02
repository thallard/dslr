import numpy as np

class LogisticRegression(object):

    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def model(self, x, theta):
        return x.dot(theta)

    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Cost function
    def cost_function(self, x, y, h):
        m = len(y)

        cost = 1 / m * np.sum(-y * np.log(h)) + (1 - y) * np.log(1 - h)
        return cost

    def gradient_descent(self, X, y, h, theta):
        gradient_value = np.dot(X.T, (h - y)) / len(y)
        theta -= self.learning_rate * gradient_value
        return theta

    def logistic_regression(self, df):
