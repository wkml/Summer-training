#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：summer_vacation 
@File    ：logistic_regression.py
@Author  ：wkml4996
@Date    ：2021/7/13 22:11 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))


class LogisticRegression(object):
    def __init__(self):
        self.theta = np.mat(0)

    # Calculating the sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Calculating the cost function
    def computecost(self, x, y, theta):
        x = x.copy()
        y = y.copy()
        # Calculate the squared term
        first = np.multiply(-y, np.log(self.sigmoid(x * theta.T)))
        #    print(first)
        second = np.multiply((1 - y), np.log(1 - self.sigmoid(x * theta.T)))
        #    print(second)
        #    regular = (lambda_ / (2 * len(X))) * np.sum(np.power(theta[:,1:],2))
        return np.sum(first - second) / (len(x))

    # Gradient descent, the incoming parameters are matrices
    def fit(self, X, y, alpha, iters):
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
            error = self.sigmoid(X @ theta.T) - y
            for j in range(parameters):
                # Record the intermediate term
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

            theta = temp
        self.theta = theta
        return theta

    def probability(self, X):
        X = X.copy()
        if not 'Ones' in X:
            X.insert(0, 'Ones', 1)
            X = np.mat(X)
        probability = self.sigmoid(X * self.theta.T)
        return probability

    # Calculate the mean square error
    def score(self, X, y, ):
        X = X.copy()
        if not 'Ones' in X:
            X.insert(0, 'Ones', 1)
            X = np.mat(X)
            y = np.mat(y).T
        probability = self.sigmoid(X * self.theta.T)
        predictions = [1 if x >= 0.5 else 0 for x in probability]
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
        accuracy = (sum(map(int, correct)) / len(correct)) * 100
        return accuracy


if __name__ == '__main__':
    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    y = data.target
    X.columns = data.feature_names
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

    LR = LogisticRegression()
    LR.fit(Xtrain, Ytrain, 0.008, 1000)
    print('accuracy = {}%'.format(LR.score(Xtest, Ytest)))

    FPR, TPR, thresholds = roc_curve(Ytest, LR.probability(Xtest))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(FPR, TPR, 'r', label='logistic')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.show()

    AUC_score = roc_auc_score(Ytest, LR.probability(Xtest))
    print('AUC score = {}'.format(AUC_score))
