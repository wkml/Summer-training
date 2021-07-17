#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ID3.py
@File    ：ID3.py
@Author  ：wkml4996
@Date    ：2021/7/15 10:50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer


class ID3Tree:
    """
    ID3分类树模型
    """

    def __init__(self):
        pass

    def calculate_entropy(self, data):
        """
        计算初始数据集中的熵
        :param data: DataFrame(parent)
        :return:
        """
        # 计算不同类型在在不同特征不同特征值中的占比
        class_total = data['result'].value_counts()
        p_k = class_total / data.shape[0]
        # p_k.dot(p_k) == (p_1)^2 + (p_2)^2 + ... + (p_k)^2
        entropy = -p_k.dot(np.log2(p_k.T))
        return entropy

    def calculate_conditional_entropy(self, data, column):
        """
        计算根据column划分后多个数据集的总熵值
        :param data: DataFrame(child)
        :param column: Performing the characteristics of the division
        :return:
        """
        # 计算不同类型在在不同特征不同特征值中的占比
        class_total = data.groupby(column)['result'].count()
        class_p = class_total / data.shape[0]

        # 划分数据集,计算条件熵
        datas = self.get_data(data, column)
        conditional_entropy = 0
        count = 0
        for data in datas.values():
            type_total = data.groupby('result')[column].count()  # 计算在该特征的情况下,各特征值中正类的占比
            p_k = type_total / data.shape[0]
            try:
                conditional_entropy += -class_p[count] * p_k.dot(np.log2(p_k.T))
            except:  # 捕捉log2(0)异常
                conditional_entropy += 0
            count += 1
        return conditional_entropy

    def get_data(self, data, column):
        """
        根据特征值划分数据
        :param data: DataFrame(X:Training data)
        :param column: Performing the characteristics of the division
        :return:
        """
        datas = {}
        for value in data[column].unique():
            datas[value] = (data[data[column] == value])
        return datas

    def create_tree(self, parents_node):
        """
        计算信息增益,并以此选择特征创建结点
        :param parents_node: Dict(tree node)
        :return: Dict(tree node)
        """
        entropy_list = []
        columns = []

        # 遍历未作为划分变量的特征,记录计算信息增益的column以及其对应信息增益
        for col in np.setdiff1d(parents_node['data'].columns[:-1], parents_node['classified']):
            columns.append(col)
            entropy_list.append(
                self.calculate_entropy(parents_node['data']) - self.calculate_conditional_entropy(parents_node['data'],
                                                                                                  col))
        # 最大信息增益小于0.15时,不再创建子结点
        try:
            if np.max(entropy_list) < 0.15:
                return -1
            else:
                # 存储每一个子结点
                nodes = []
                # 定位划分数据集后信息增益最大的特征
                column = columns[int(np.argmax(entropy_list))]
                # 遍历该特征的唯一值列表创建子结点
                for value in parents_node['data'][column].unique():
                    # 根据特征值划分数据集
                    node = {'data': parents_node['data'][parents_node['data'][column] == value]}
                    # 记录当前结点是根据哪一特征创建出来的
                    node['column'] = column
                    # 记录已经作为划分变量的特征
                    node['classified'] = parents_node['classified'] + [column]
                    # 记录当前结点当前划分特征的特征值
                    node['value'] = value
                    # 获取占比最大的类作为该结点的类
                    node['label'] = parents_node['data']['result'].value_counts().sort_values(ascending=False).index[
                        0]
                    # 创建子结点
                    node['child'] = self.create_tree(node)
                    nodes.append(node)
            # 返回子结点
            return nodes
        except:
            return -1

    def judge_label(self, test_data, node):
        """
        根据测试数据的某个特征将该样本划分出类别
        :param test_data: DataFrame(Xtest)
        :param node: TREE
        :return:
        """
        # 当为叶结点时,返回当前结点
        if node['child'] == -1:
            return node['label']
        else:
            # 遍历当前结点的子结点
            for child_node in node['child']:
                if child_node['value'] == test_data[child_node['column']]:
                    return self.judge_label(test_data, child_node)
            # 当测试数据出现了训练数据中没有出现过的特征值,返回当前结点
            return node['label']

    def recall(self):
        """
        计算召回率,评估模型
        :return: recall rate
        """
        self.recall_percent = 100 * self.correct_labels[self.correct_labels == 1].shape[0] / \
                              self.test_label[self.test_label == 1].shape[0]

    def correct(self):
        """
        计算正确率,评估模型
        :return: correct rate
        """
        self.correct_labels = self.test_label[self.test_label == self.predict_labels]
        self.correct_percent = 100 * self.correct_labels.shape[0] / self.test_label.shape[0]

    def fit(self, train_data, train_label):
        """
        连接数据,创建根结点,开始创建树
        :param train_data: DataFrame(Xtrain)
        :param train_label: DataFrame(Ytrain)
        :return:
        """
        data = pd.concat([train_data, train_label], axis=1)
        self.tree = {'data': data, 'classified': ['result'], 'column': 'result'}
        self.tree['child'] = self.create_tree(self.tree)

    def predict(self, test_data, test_label):
        """
        预测待预测数据的标签,评估模型
        :param test_data: Dataframe(Xtest)
        :param test_label: Dataframe(Ytest)
        :return:
        """
        # 转换标签数据类型
        self.test_label = np.array(test_label).ravel()
        predict_label = []
        # 遍历测试样本,给样本分类
        for i in range(test_data.shape[0]):
            label = self.judge_label(test_data.iloc[i], self.tree)
            predict_label.append(label)
        # 转换标签数据类型
        self.predict_labels = np.array(predict_label).ravel()
        # 计算正确率
        self.correct()
        # 计算召回率
        self.recall()
        print("正确率为：", self.correct_percent)
        print("召回率为：", self.recall_percent)


if __name__ == '__main__':
    tree = ID3Tree()

    # 数据预处理，Tree难以处理连续性数据，先进行分箱
    # est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    # data = load_breast_cancer()
    # X = pd.DataFrame(data.data)
    # X = pd.DataFrame(est.fit_transform(X))
    # y = pd.DataFrame(data.target)
    # X.columns = data.feature_names
    # y.columns = ['result']
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=1)

    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    data = pd.read_csv('lenses.txt', delim_whitespace=True, index_col=0)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    y.name = 'result'
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=1)

    tree.fit(Xtrain, Ytrain)
    tree.predict(Xtest, Ytest)
