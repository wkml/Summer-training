#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：Adaboost.py
@Author  ：wkml4996
@Date    ：2021/8/4 21:14 
"""
import numpy as np
import math


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
        arr[datam[:, dimen] > threshval] = 1.0
    # 返回值为更新后的类别
    return arr


def bulidStump(dataarr, classarr, d):  # 数据集，类别列表，权值d
    """
    生成树桩
    :param dataarr: 数据集X
    :param classarr: Y
    :param d: 权重
    :return:
    """
    datam = np.mat(dataarr)
    labelm = np.mat(classarr).T
    m, n = datam.shape
    # 走几步
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    # 最小误差
    minError = math.inf
    # 对每一个特征遍历
    for i in range(n):
        # 获取该列特征的最大值和最小值
        rangeMin = datam[:, i].min()
        rangeMax = datam[:, i].max()
        # 确定一步多长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 对每个步长进行遍历，范围[-1，int(numSteps)]
        for j in range(-1, int(numSteps) + 1):
            # 每个不等式遍历
            for inequal in ['lt', 'rt']:
                # 计算阈值，上面的j从-1开始可以保证当前分类把所有数据集都分成一个类别
                threshval = (rangeMin + float(j) * stepSize)
                # 根据阈值计算预测值
                predictedVal = stumpClassify(datam, i, threshval, inequal)
                # 误差数组，初始化全1
                errArr = np.mat(np.ones((m, 1)))
                # 把误差数组中的预测值和实际值相等的值置为0
                errArr[predictedVal == labelm] = 0
                # 计算整体误差
                weithtedError = d.T * errArr
                print('dim:%d, thresh:%.2f, thresh ineqal:%s, weighted error:%.3f' % (
                    i, threshval, inequal, weithtedError))
                # 当最终误差小于当下最小值时更新相关变量
                if weithtedError < minError:
                    minError = weithtedError
                    bestClassEst = predictedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshval
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDs(dataarr, classl, numit=40):  # 数据集，类别列表，迭代次数
    """
    :param dataarr: 数据集X
    :param classl: y
    :param numit: 迭代次数
    :return:
    """
    weakClassArr = []  # 决策数组
    m = dataarr.shape[0]  # 特征数目
    d = np.mat(np.ones((m, 1)) / m)  # 权值数组
    aggClassEst = np.mat(np.zeros((m, 1)))  # 类别估计累计值
    for i in range(numit):
        bestStump, error, classEst = bulidStump(dataarr, classl, d)  # 获取当前的最佳单层决策树
        # print('d:',d.T)
        if error > 0.5:
            break
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算alpha值，使用max确保在没有错误的时候不会发生零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 将决策树加入到数组中
        # print('classEst:',classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classl).T,
                            classEst)  # np.multiply对应元素相乘，如果和预测值相同，结果是1，再乘以-alpha，结果是-alpha，反之若不同，结果是alpha，再将最终结果转换成矩阵形式
        # print('expon:',expon.T)
        d = np.multiply(d, np.exp(expon))  # 更新d值
        d = d / d.sum()
        aggClassEst += alpha * classEst  # 更新累计类别估计值
        # print('aggClassEst:',aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classl).T,
                                np.ones((m, 1)))  # 使用sign将大于0的设置为1，小于0的设置为-1，再和实际类别异或，计算错误率
        # print('sign:',np.sign(aggClassEst)!=np.mat(classl).T)
        # print(aggErrors)
        errorRate = aggErrors.sum() / m
        # print('erroe:',errorRate)
        if errorRate == 0.0: break  # 错误率为0时退出循环
    return weakClassArr


def adaClassify(dataarr, classifier):
    datam = np.mat(dataarr)
    m = datam.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier)):
        classEst = stumpClassify(datam, classifier[i]['dim'], classifier[i]['thresh'], classifier[i]['ineq'])
        aggClassEst += classifier[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    pass
