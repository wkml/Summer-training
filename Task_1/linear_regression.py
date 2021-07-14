#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：summer_vacation 
@File    ：linear_regression.py
@Author  ：wkml4996
@Date    ：2021/7/13 17:16 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))


class LinearRegression(object):
    def __init__(self):
        self.theta = np.mat(0)

    # Calculating the cost function
    def computecost(self, x, y, theta):
        # Calculate the squared term
        inner = np.power(((x @ theta.T) - y), 2)
        return np.sum(inner) / (2 * x.shape[0])

    # Gradient descent, the incoming parameters are matrices
    def gradient_descent(self, X, y, alpha, iters):
        X = X.copy()
        X.insert(0, 'Ones', 1)
        X = np.mat(X)
        y = np.mat(y).T
        theta = np.mat(np.zeros(X.shape[1]))
        temp = np.matrix(np.zeros(theta.shape))
        # Number of features
        parameters = int(theta.ravel().shape[1])

        # iters sub iterations
        for i in range(iters):
            error = (X @ theta.T) - y
            for j in range(parameters):
                # Record the intermediate term
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

            theta = temp
        self.theta = theta
        return theta

    # Least squares method with incoming parameters as matrix
    def fit(self, X, y):
        X = X.copy()
        X.insert(0, 'Ones', 1)
        X = np.mat(X)
        y = np.mat(y).T
        # X.T@X equivalent to X.T.dot(X)
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.theta = theta.T
        return theta.T

    # Calculate the mean square error
    def score(self, X, y, ):
        X = X.copy()
        X.insert(0, 'Ones', 1)
        X = np.mat(X)
        y = np.mat(y).T
        inner = np.power(((X @ self.theta.T) - y), 2)
        return np.sum(inner) / (X.shape[0])

    # Calculate the root mean square error
    def RMSE(self, x, y, theta):
        return np.sqrt(self.score(x, y, ))

    # Data Visualization
    def visualization(self, x, y):
        pass


if __name__ == '__main__':
    # Data pre-processing
    data = load_diabetes()
    X = pd.DataFrame(data.data)
    y = data.target
    X.columns = data.feature_names
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

    # Model Selection
    LN = LinearRegression()
    theta1 = LN.gradient_descent(Xtrain, Ytrain, 0.0093, 1000)
    print('The Mean Square Error of gradient descent is {:.5}'.format(LN.score(Xtest, Ytest)))
    theta2 = LN.fit(Xtrain, Ytrain)
    print('The Mean Square Error of least squares method is {:.5}'.format(LN.score(Xtest, Ytest)))
