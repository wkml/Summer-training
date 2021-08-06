#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：Adaboost.py
@Author  ：wkml4996
@Date    ：2021/8/4 21:14 
"""
import numpy as np
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def stumpClassify(datam, dimen, threshval, threshineq):
    """
    分类
    :param datam: 数据集X
    :param dimen: Y
    :param threshval: 阈值
    :param threshineq: 大小关系
    :return:
    """
    arr = np.ones((datam.shape[0], 1))
    if threshineq == 'lt':
        arr[datam[:, dimen] <= threshval] = -1.0
    else:
        arr[datam[:, dimen] > threshval] = -1.0
    # 返回值为更新后的类别
    return arr


def bulidStump(Xtrain, Ytrain, d):  # 数据集，类别列表，权值d
    """
    生成树桩
    :param Xtrain: 数据集X
    :param Ytrain: Y
    :param d: 权重
    :return:
    """
    Xtrain = np.mat(Xtrain)
    Ytrain = np.mat(Ytrain).T
    sanple_num, feature_num = Xtrain.shape
    # 走几步
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((sanple_num, 1)))
    # 最小误差
    minError = math.inf
    # 对每一个特征遍历
    for i in range(feature_num):
        # 获取该列特征的最大值和最小值
        rangeMin = Xtrain[:, i].min()
        rangeMax = Xtrain[:, i].max()
        # 确定一步多长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 对每个步长进行遍历，范围[-1，int(numSteps)]
        for j in range(-1, int(numSteps) + 1):
            # 每个不等式遍历
            for inequal in ['lt', 'rt']:
                # 计算阈值，上面的j从-1开始可以保证当前分类把所有数据集都分成一个类别
                threshval = (rangeMin + float(j) * stepSize)
                # 根据阈值计算预测值
                predictedVal = stumpClassify(Xtrain, i, threshval, inequal)
                # 误差数组，初始化全1
                errArr = np.mat(np.ones((sanple_num, 1)))
                # 把误差数组中的预测值和实际值相等的值置为0
                errArr[predictedVal == Ytrain] = 0
                # 计算整体误差
                weithtedError = d.T * errArr

                # 当最终误差小于当下最小值时更新相关变量
                if weithtedError < minError:
                    minError = weithtedError
                    bestClassEst = predictedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshval
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDs(Xtrain, Ytrain, numit=40):  # 数据集，类别列表，迭代次数
    """
    :param Xtrain: 数据集X
    :param Ytrain: y
    :param numit: 迭代次数
    :return:
    """
    # 决策数组
    weakClassArr = []
    # 样本数目
    sample_num = Xtrain.shape[0]
    # 权值数组
    d = np.mat(np.ones((sample_num, 1)) / sample_num)
    for i in range(numit):
        bestStump, error, classEst = bulidStump(Xtrain, Ytrain, d)  # 获取当前的最佳单层决策树
        # print('d:',d.T)
        if error > 0.5:
            break
        # print(error)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算alpha值，使用max确保在没有错误的时候不会发生零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 将决策树加入到数组中
        # print('classEst:',classEst.T)
        # np.multiply对应元素相乘，如果和预测值相同，结果是1，再乘以-alpha，结果是-alpha，反之若不同，结果是alpha，再将最终结果转换成矩阵形式
        expon = np.multiply(-1 * alpha * np.mat(Ytrain).T,
                            classEst)
        # print('expon:',expon.T)
        # 更新d值
        d = np.multiply(d, np.exp(expon))
        d = d / d.sum()
    return weakClassArr


def adaClassify(dataarr, classifier):
    datam = np.mat(dataarr)
    m = datam.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier)):
        classEst = stumpClassify(datam, classifier[i]['dim'], classifier[i]['thresh'], classifier[i]['ineq'])
        aggClassEst += classifier[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    clf = DecisionTreeClassifier()
    boost = AdaBoostClassifier()

    data = load_breast_cancer()
    X = data.data
    y = data.target
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] = -1
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y)

    clf.fit(Xtrain, Ytrain)
    boost.fit(Xtrain, Ytrain)
    model = adaBoostTrainDs(Xtrain, Ytrain, numit=15)
    Adscore = Ytest[np.array(adaClassify(Xtest, model)).ravel() == Ytest].shape[0] / Ytest.shape[0]

    print('Decision Tree: %f;' % clf.score(Xtest, Ytest), 'Sklearn Adaboost: %f;' % boost.score(Xtest, Ytest),
          'Self Adaboost: %f' % Adscore)
