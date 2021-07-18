#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：linear_regression.py 
@File    ：bayes.py
@Author  ：wkml4996
@Date    ：2021/7/17 10:26 
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的dataframe
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''
        # 训练集样本总数
        row_num = feature.shape[0]
        # 训练集属性总数
        columns = feature.columns
        # 种类数
        unique_label_count = len(set(label))

        for c in label:
            if c in self.label_prob:
                self.label_prob[c] += 1
            else:
                self.label_prob[c] = 1

        for C in self.label_prob.keys():
            # 计算每种类别在数据集中出现的概率，拉普拉斯平滑
            self.label_prob[C] += 1
            self.label_prob[C] /= (unique_label_count + row_num)

            # 构建self.condition_prob中的key
            self.condition_prob[C] = {}
            # 特征数量
            for col in columns:
                self.condition_prob[C][col] = {}
                # k 为 特征所有可能取值，为了进行邮件处理，只考虑出现与否不考虑次数
                for k in np.unique(feature.loc[:, col], axis=0):
                    # 这里可以改一下，for循环可以去掉，这一步主要还是对拉普拉斯平滑的初始化，由于特征都是1或0，for循环没什么意义
                    self.condition_prob[C][col][0] = 1
                    self.condition_prob[C][col][1] = 1

        for i in range(feature.shape[0]):
            # label[i] == i的标签
            c = label[i]
            for col, j in zip(columns, range(len(columns))):
                k = feature.iloc[i, j]
                if k == 0:
                    self.condition_prob[c][col][0] += 1
                else:
                    self.condition_prob[c][col][1] += 1

        for C in self.condition_prob.keys():
            for k in self.condition_prob[C].keys():
                # 拉普拉斯平滑
                N_k = len(self.condition_prob[C][k].keys())  # == 2
                total = N_k + label[label == C].shape[0]
                for inner_key in self.condition_prob[C][k].keys():
                    # 计算每种类别确定的条件下各个特征出现的概率
                    self.condition_prob[C][k][inner_key] /= total

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的DataFrame
        :return:
        '''

        result = []
        # 对每条测试数据都进行预测
        for i in range(feature.shape[0]):
            fea_i = feature.iloc[i, :]
            # 可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                # 计算概率
                prob[ii] = label_prob
                for col in feature.columns:
                    try:
                        # 这里的try是为了针对出现了训练集中没出现过的词的情况
                        if fea_i[col] == 0:
                            prob[ii] *= self.condition_prob[label][col][0]
                        else:
                            prob[ii] *= np.power(self.condition_prob[label][col][1], fea_i[col])
                    except:
                        continue

                ii += 1
            # 取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)

    def score(self, Xtest, Ytest):
        """
        返回贝叶斯的分数
        :param Xtest: ndarray
        :param Ytest: ndarray
        :return: score
        """
        predict = self.predict(Xtest)
        correct_labels = predict[predict == Ytest]
        score = correct_labels.shape[0] / predict.shape[0] * 100
        print('the score of bayes is {}'.format(score))


if __name__ == '__main__':
    data = load_breast_cancer()
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X = pd.DataFrame(est.fit_transform(data.data), columns=data.feature_names)
    y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, )

    BY = NaiveBayesClassifier()
    BY.fit(X_train, Y_train)
    BY.score(X_test, Y_test)
