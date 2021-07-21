#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bayes_project2.py 
@File    ：SVM.py
@Author  ：wkml4996
@Date    ：2021/7/20 21:31 
"""
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC


class SVM:
    def __init__(self, data, labels, C=1.0, toler=0.001, kernel_option='linear', max_iter=50):
        self.Xtrain = np.mat(data)  # each row stands for a sample
        self.Ytrain = np.mat(labels).T  # corresponding label
        self.C = C  # slack variable
        self.toler = toler  # termination condition for iteration
        self.num_samples = self.Xtrain.shape[0]  # number of samples
        self.alphas = np.mat(np.zeros((self.num_samples, 1)))  # Lagrange factors for all samples
        self.b = 0
        self.max_iter = max_iter
        self.errorCache = np.mat(np.zeros((self.num_samples, 2)))
        self.kernelOpt = kernel_option
        self.kernelMat = self.calc_kernel_matrix()

    def calc_kernel_matrix(self):
        """

        :return:
        """
        num_samples = self.Xtrain.shape[0]
        kernel_matrix = np.mat(np.zeros((num_samples, num_samples)))
        for i in range(num_samples):
            kernel_matrix[:, i] = self.calc_kernel_value(self.Xtrain[i, :])
        return kernel_matrix

    def calc_kernel_value(self, sample_x, ):
        """
        计算矩阵里的值
        :param matrix_x:
        :param sample_x:
        :param kernel_option:
        :return:
        """
        num_samples = self.Xtrain.shape[0]
        kernel_value = np.mat(np.zeros((num_samples, 1)))

        if self.kernelOpt is 'linear':
            self.kernelOpt = self.Xtrain * sample_x.T
        elif self.kernelOpt is 'rbf':
            sigma = 1.0
            for i in range(num_samples):
                diff = self.Xtrain[i, :] - sample_x
                kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))
        return kernel_value

    def calc_error(self, i):
        fuc_x = float(np.multiply(self.alphas, self.Ytrain).T * (self.Xtrain * self.Xtrain[i, :].T)) + self.b
        error = fuc_x - float(self.Ytrain[i])
        return error

    def selectAlpha_j(self, i, error_i):
        self.errorCache[i] = [1, error_i]  # mark as valid(has been optimized)
        candidate_alpha_list = np.nonzero(self.errorCache[:, 0].A)[0]  # mat.A return array
        max_step = 0
        j = 0
        error_j = 0

        # find the alpha with max iterative step
        if len(candidate_alpha_list) > 1:
            for alpha_k in candidate_alpha_list:
                if alpha_k == i:
                    continue
                error_k = self.calc_error(alpha_k)
                if abs(error_k - error_i) > max_step:
                    max_step = abs(error_k - error_i)
                    j = alpha_k
                    error_j = error_k
        # if came in this loop first time, we select alpha j randomly
        else:
            j = i
            while j == i:
                j = int(random.uniform(0, self.num_samples))
            error_j = self.calc_error(j)

        return j, error_j

    def SMO(self):
        dataMatrix = self.Xtrain
        labelMat = self.Ytrain
        max_iter = self.max_iter
        # 初始化b参数，统计dataMatrix的维度
        m, n = np.shape(dataMatrix)

        # 初始化迭代次数
        iter_num = 0

        # 最多迭代matIter次
        while iter_num < max_iter:
            alpha_pairs_changed = 0  # 用于记录alpha是否已经进行优化
            # 启发式选择 外循环  x_1 to x_m
            for i in range(m):
                # 步骤1：计算误差Ei
                Ei = self.calc_error(i)
                # 优化alpha，更设定一定的容错率。
                if ((labelMat[i] * Ei < -self.toler) and (self.alphas[i] < self.C)) or (
                        (labelMat[i] * Ei > self.toler) and (self.alphas[i] > 0)):
                    # 启发式选择，内循环
                    # 步骤1：计算误差Ej
                    j, Ej = self.selectAlpha_j(i, Ei)
                    # 保存更新前的aplpha值，使用深拷贝
                    alphaIold = self.alphas[i].copy()
                    alphaJold = self.alphas[j].copy()
                    # 步骤2：计算上下界L和H
                    if labelMat[i] != labelMat[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    if L == H:
                        # print("样本:%d L==H" % (i))
                        continue

                    # 步骤3：计算eta，即η。 x_i**2+x_j**2-2x_i*x_j
                    eta = dataMatrix[i, :] * dataMatrix[i, :].T + dataMatrix[j, :] * dataMatrix[j,
                                                                                     :].T - 2.0 * dataMatrix[
                                                                                                  i,
                                                                                                  :] * dataMatrix[
                                                                                                       j, :].T
                    if eta == 0:
                        continue
                    # 步骤4：更新alpha_j
                    self.alphas[j] += labelMat[j] * (Ei - Ej) / eta
                    # 步骤5：修剪alpha_j
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    if self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(self.alphas[j] - alphaJold) < 0.00001:
                        # print("样本:%d alpha_j变化太小" % (i))
                        continue
                    # 步骤6：更新alpha_i
                    self.alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - self.alphas[j])
                    # 步骤7：更新b_1和b_2
                    b1 = self.b - Ei - labelMat[i] * (self.alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i,
                                                                                                       :].T - \
                         labelMat[
                             j] * (self.alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                    b2 = self.b - Ej - labelMat[i] * (self.alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j,
                                                                                                       :].T - \
                         labelMat[
                             j] * (self.alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                    # 步骤8：根据b_1和b_2更新b
                    if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                        self.b = b1
                    elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    # 统计优化次数
                    alpha_pairs_changed += 1
                    # 打印统计信息
                    print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alpha_pairs_changed))
            # 更新迭代次数
            if alpha_pairs_changed == 0:
                iter_num += 1
            else:
                iter_num = 0
            print("迭代次数: %d" % iter_num)

    def get_w(self):
        dataMat, labelMat = np.array(self.Xtrain), np.array(self.Ytrain)
        w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, dataMat.shape[1])) * dataMat).T, self.alphas)
        return w

    def predict(self, Xtest):
        Xtest = np.mat(Xtest)
        w = self.get_w().reshape(1, self.Xtrain.shape[1])
        h = (w * Xtest.T + self.b).A.ravel()
        for i in range(h.shape[0]):
            h[i] = 1 if h[i] > 0 else -1
        return h

    def score(self, Xtest, Ytest):
        predict = self.predict(Xtest)
        correct_labels = predict[predict == Ytest]
        score = correct_labels.shape[0] / predict.shape[0] * 100
        print('the score of SVM is {}'.format(score))


def selectJrand(i, m):
    """

    :param i: alpha
    :param m: alpha参数个数
    :return:
    """
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def showClassifer(dataMat, w, b):
    # 绘制样本点
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:  # 如果alpha>0,表示 alpha所在的不等式条件起作用了
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    data = load_breast_cancer()
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X = pd.DataFrame(est.fit_transform(data.data), columns=data.feature_names)
    print(X.shape)
    y = data.target
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] = -1
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
    svm = SVM(Xtrain, Ytrain, C=0.6)
    svm.SMO()
    svm.score(Xtest, Ytest)
    b, alphas = svm.b, svm.alphas
    w = svm.get_w()
    # print("w:", w)
    # showClassifer(dataMat, w, b)
