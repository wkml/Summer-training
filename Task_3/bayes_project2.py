#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bayes_project.py 
@File    ：bayes_project2.py
@Author  ：wkml4996
@Date    ：2021/7/17 22:12 
"""
from sklearn.model_selection import train_test_split

from bayes import NaiveBayesClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os


if __name__ == '__main__':

    txt_X = []
    path = "F:/notebooks/summer_train/bin/ham"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:
        with open(path + '/' + file, encoding='gbk') as f:
            iter_f = iter(f)
            lines = ""
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                lines = lines + line
            txt_X.append(lines)
    txt_Y1 = ['ham'] * len(txt_X)
    path = "F:/notebooks/summer_train/bin/spam"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:
        with open(path + '/' + file, encoding='gbk') as f:
            iter_f = iter(f)
            lines = ""
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                lines = lines + line
            txt_X.append(lines)
    txt_Y2 = ['spam'] * (len(txt_X) - len(txt_Y1))
    txt_Y = np.array(txt_Y1 + txt_Y2)


    vec = CountVectorizer()
    txt_X = vec.fit_transform(txt_X)
    txt_X = pd.DataFrame(txt_X.toarray(),columns=vec.get_feature_names())

    X_train, X_test, Y_train, Y_test = train_test_split(txt_X, txt_Y, test_size=0.3, )

    BY = NaiveBayesClassifier()
    BY.fit(X_train, Y_train)
    BY.score(X_test, Y_test)