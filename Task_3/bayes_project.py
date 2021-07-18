#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：bayes.py 
@File    ：bayes_project.py
@Author  ：wkml4996
@Date    ：2021/7/17 14:53 
"""
from sklearn.datasets import fetch_20newsgroups
from bayes import NaiveBayesClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    categories = ["sci.space"  # 科学技术 - 太空
        #, "rec.sport.hockey"  # 运动 - 曲棍球
        #, "talk.politics.guns"  # 政治 - 枪支问题
        , "talk.politics.mideast"  # 政治 - 中东问题
                  ]
    train = fetch_20newsgroups(subset="train", categories=categories)
    test = fetch_20newsgroups(subset="test", categories=categories)
    Xtrain = train.data
    Xtest = test.data
    Ytrain = train.target
    Ytest = test.target

    vec = CountVectorizer()
    Xtrain = pd.DataFrame(vec.fit_transform(Xtrain).toarray(), columns=vec.get_feature_names())
    Xtest = pd.DataFrame(vec.fit_transform(Xtest).toarray(), columns=vec.get_feature_names())

    clf = NaiveBayesClassifier()
    clf.fit(Xtrain, Ytrain)
    clf.score(Xtest, Ytest)
