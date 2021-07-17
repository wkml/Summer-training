#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ID3.py
@File    ：CART.py
@Author  ：wkml4996
@Date    ：2021/7/15 13:13 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import KBinsDiscretizer


class CARTTree:
    """
    CART分类树模型
    """
    def __init__(self):
        pass

    def calculate_original_gini(self, data):
        """
        计算初始数据集中的基尼指数
        :param data:
        :return:
        """
        # 计算不同类型在在不同特征不同特征值中的占比
        class_total = data['result'].value_counts()
        p_k = class_total / data.shape[0]

        # 计算基尼指数
        original_gini = 1 - p_k.dot(p_k.T)
        return original_gini

    def calculate_processing_gini(self, data, column):
        """
        计算最优划分变量及最优切分点
        :param data:
        :param column:
        :return:
        """
        # 根据当前特征划分数据集
        datas = self.get_data(data, column)

        # 计算并存储各特征值基尼指数
        ginis = {}
        for value in datas.values():
            # 计算在该特征的情况下,column中正负类的占比
            type_total = value.groupby('result')[column].count()
            type_p = type_total / value.shape[0]
            gini = (value.shape[0] / data.shape[0]) * (1 - type_p.dot(type_p.T)) + (
                    1 - value.shape[0] / data.shape[0]) * (1 - (1 - type_p).dot((1 - type_p).T))
            ginis[gini] = value[column].unique()[0]
        return min(ginis.keys()), ginis[min(ginis.keys())]

    def get_data(self, data, column):
        """
        根据特征值划分数据
        :param data:
        :param column:
        :return:
        """
        datas = {}
        for value in data[column].unique():
            datas[value] = (data[data[column] == value])
        return datas

    def divide_data(self, data, column, a):
        """
        根据当前最优划分变量的最优划分点划分数据集
        :param data:
        :param column:
        :param a:
        :return:
        """
        datas = {}
        datas['N1'] = data[data[column] == a]
        datas['N2'] = data[data[column] != a]
        return datas

    def create_tree(self, parents_node):
        """
        计算基尼指数,并以此选择特征创建结点
        :param parents_node:
        :return:
        """
        gini_list = []
        columns = []
        value = []
        # 遍历未作为划分变量的特征,记录计算基尼指数的column以及其对应基尼指数
        for col in np.setdiff1d(parents_node['data'].columns[:-1], parents_node['classified']):
            columns.append(col)
            processing_gini, a = self.calculate_processing_gini(parents_node['data'], col)
            gini_list.append(self.calculate_original_gini(parents_node['data']) - processing_gini)
            value.append(a)
        try :
            if np.max(gini_list) < 0.15:  # 最大基尼指数小于0.15时,不再创建子结点
                return -1
            else:
                nodes = []  # 存储每一个结点
                column = columns[int(np.argmax(gini_list))]  # 定位划分数据集后基尼指数的特征
                a = value[int(np.argmax(gini_list))]
                for data in self.divide_data(parents_node['data'], column, a).values():  # 遍历该特征的唯一值列表创建子结点
                    node = {'data': data}  # 根据特征值划分数据集
                    node['column'] = column  # 根据特征值划分数据集
                    node['classified'] = parents_node['classified'] + [column]  # 记录已经作为划分变量的特征
                    node['a'] = data[column].unique()  # 记录当前结点当前划分特征的特征值
                    node['label'] = data['result'].value_counts().sort_values(ascending=False).index[
                        0]  # 获取占比最大的类作为该结点的类
                    node['child'] = self.create_tree(node)  # 创建子结点
                    nodes.append(node)
            return nodes  # 返回子结点
        except:
            return -1

    def judge_label(self, test_data, node):
        """
        根据测试数据的某个特征将该样本划分出类别
        :param test_data:
        :param node:
        :return:
        """
        if node['child'] == -1:  # 当为叶结点时,返回当前结点
            return node['label']
        else:
            if test_data[node['child'][0]['column']] in node['child'][0]['a']:
                return self.judge_label(test_data, node['child'][0])
            elif test_data[node['child'][0]['column']] in node['child'][1]['a']:
                return self.judge_label(test_data, node['child'][1])
            return node['label']  # 当测试数据出现了训练数据中没有出现过的特征值,返回当前结点

    def recall(self):
        """
        计算召回率,评估模型
        :return:
        """
        self.recall_percent = 100 * self.correct_labels[self.correct_labels == 1].shape[0] / \
                              self.test_label[self.test_label == 1].shape[0]

    def correct(self):
        """
        计算正确率,评估模型
        :return:
        """
        self.correct_labels = self.test_label[self.test_label == self.predict_labels]
        self.correct_percent = 100 * self.correct_labels.shape[0] / self.test_label.shape[0]

    def fit(self, train_data, train_label):
        """
        连接数据,创建根结点,开始创建树
        :param train_data:
        :param train_label:
        :return:
        """
        data = pd.concat([train_data, train_label], axis=1)  # 连接数据
        self.tree = {'data': data, 'classified': ['result'], 'column': 'result'}  # 创建根结点
        self.tree['child'] = self.create_tree(self.tree)

    def predict(self, test_data, test_label):
        """
        预测待预测数据的标签,评估模型
        :param test_data:
        :param test_label:
        :return:
        """
        self.test_label = np.array(test_label).ravel()  # 转换标签数据类型
        predict_label = []
        for i in range(test_data.shape[0]):  # 遍历测试样本,给样本分类
            label = self.judge_label(test_data.iloc[i], self.tree)
            predict_label.append(label)
        self.predict_labels = np.array(predict_label)  # 转换标签数据类型
        self.correct()  # 计算正确率
        self.recall()  # 计算召回率
        print("正确率为：", self.correct_percent)
        # print("召回率为：", self.recall_percent)


if __name__ == '__main__':
    tree = CARTTree()

    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    # data = pd.read_csv('lenses.txt',delim_whitespace = True,index_col=0)
    # X = data.iloc[:,1:-1]
    # y = data.iloc[:,-1]
    # y.name = 'result'
    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    X = pd.DataFrame(est.fit_transform(X))
    y = pd.DataFrame(data.target)
    X.columns = data.feature_names
    y.columns = ['result']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=1)

    tree.fit(Xtrain, Ytrain)
    tree.predict(Xtest, Ytest)
