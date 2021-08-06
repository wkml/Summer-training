#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：k_means.py
@Author  ：wkml4996
@Date    ：2021/8/6 14:11 
"""
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle


def calc_dis(x_train, center, k):
    """
    计算欧式距离
    :param x_train: 训练集
    :param center: 质心
    :param k: 簇数
    :return:
    """
    dis_list = []
    for data in x_train:
        # 相减
        diff = np.tile(data, (k, 1)) - center
        distance = np.sum(diff ** 2, axis=1) ** 0.5  # 和  (axis=1表示行)
        dis_list.append(distance)
    # 返回一个每个点到质点的距离
    return np.array(dis_list)


def calc_center(x_train, center, k):
    """
    计算簇心
    :param x_train:
    :param center:
    :param k:
    :return:center,changed
    """
    distance = calc_dis(x_train, center, k)
    min_distance = np.argmin(distance, axis=1)
    new_center = pd.DataFrame(x_train).groupby(min_distance).mean().values
    changed = new_center - center
    return changed, new_center


def k_means(x_train, k, iter_num=50):
    """
    k_means
    :param x_train: ndarray,训练集
    :param k: 簇数
    :param iter_num:迭代次数
    :return: model
    """
    center = random.sample(x_train.tolist(), k)

    for i in range(iter_num):
        changed, center = calc_center(x_train, center, k)
        if np.any(changed) < 1e-3:
            break
    center = sorted(center.tolist())

    cluster = []
    distance = calc_dis(x_train, center, k)
    min_distance = np.argmin(distance, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(min_distance):
        cluster[j].append(x_train[i])

    return center, cluster, min_distance


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = MinMaxScaler().fit_transform(X)
    # suffer数据
    # X, y = shuffle(X,y)
    center, cluster, y_pre = k_means(X, np.unique(y).shape[0])
    # 降维
    tsne = TSNE(n_components=2, init='random', random_state=177).fit(X)
    # 可视化
    df = pd.DataFrame(tsne.embedding_)
    df['labels'] = y_pre
    df1 = df[df['labels'] == 0]
    df2 = df[df['labels'] == 1]
    df3 = df[df['labels'] == 2]
    # 绘制画布
    fig = plt.figure(figsize=(9, 6))
    plt.plot(df1[0], df1[1], 'bo', df2[0], df2[1], 'r*', df3[0], df3[1], 'gD')
    plt.show()
