import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# class LogisticRegression(object):
#
#     def __init__(self, iterations, learning_rate):
#         self.iterations = iterations
#         self.learning_rate = learning_rate
#         self.cost_history = []
#         self.theta = []
#
#     def model(self, x, theta):
#         return x.dot(theta)
#
#     # Sigmoid function
#     def sigmoid(self, x):
#         value =  1 / (1 + np.exp(-x))
#         return value.reshape(value.shape[0], 1)
#
#     # Cost function
#     def cost_function(self, x, y, h):
#         m = len(y)
#         # cost = 1 / m * np.sum(-y * np.log(h)) + (1 - y) * np.log(1 - h)
#         cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
#         return cost
#
#     # Gradient descent
#     def gradient_descent(self, X, y, h, theta):
#         gradient_value = np.dot(X.T, (h - y)) / len(y)
#         theta -= self.learning_rate * gradient_value
#         return theta
#
#     # Main function with loop gradient descent
#     def logistic_regression(self, X, y):
#         X = np.insert(X, 0, 1, axis=1)
#
#         for i in np.unique(y):
#             y_onevsall = np.where(y == i, 1, 0)
#             theta = np.zeros(X.shape[1])
#             theta = theta.reshape(theta.shape[0], 1)
#             print(theta.shape)
#             cost = []
#             for j in range(self.iterations):
#                 z = X.dot(theta)
#                 h = self.sigmoid(z)
#                 print(h.shape)
#                 theta = self.gradient_descent(X, y_onevsall, h, theta)
#                 cost.append(self.cost_function(X, y_onevsall, theta))
#             self.theta.append((theta, i))
#             self.cost.append((cost, i))
#         return self
#
#
#
# data = pd.read_csv('../datasets/dataset_train.csv')
# data.replace('', np.NaN, inplace=True)
# data.dropna(inplace=True)
# data.reset_index(inplace=True)
#
# data_T = data.T
# y_data = data['Hogwarts House'].values
#
# X = data.drop(['Index', 'Hogwarts House', 'Best Hand', 'First Name', 'Last Name', 'Birthday'], axis=1).values
#
# # print(X)
# # print(y_data)
# X_train,X_test,y_train,y_test = train_test_split(X,y_data,test_size = 0.33)
# logi = LogisticRegression(1000, 0.05).logistic_regression(X_train, y_train)