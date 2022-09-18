# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:27:34 2022

@author: 18721
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (10, 6)  # 设置输出图片大小


'''预先任务：加载数据并分割测试集和训练集'''
from sklearn.model_selection import train_test_split
# 一般先取出X和y后再切割，有些情况会使用到未切割的，这时候X和y就可以用,x是清洗好的数据，y是我们要预测的存活数据'Survived'
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
data = pd.read_csv(path+'clean_data.csv')
train = pd.read_csv(path+'train.csv')
X = data
y = train['Survived']
# 对数据集进行切割
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# 默认参数逻辑回归模型
lr = LogisticRegression)
lr.fit(X_train, y_train)


'''模型评估'''
# 交叉验证（cross-validation）是一种评估泛化性能的统计学方法，它比单次划分训练集和测试集的方法更加稳定、全面。
# 在交叉验证中，数据被多次划分，并且需要训练多个模型。
# 最常用的交叉验证是 k 折交叉验证（k-fold cross-validation），其中 k 是由用户指定的数字，通常取 5 或 10。
# 准确率（precision）度量的是被预测为正例的样本中有多少是真正的正例
# 召回率（recall）度量的是正类样本中有多少被预测为正类
# f-分数是准确率与召回率的调和平均
'''任务一：交叉验证'''
# 用10折交叉验证来评估之前的逻辑回归模型
# 计算交叉验证精度的平均值
# 交叉验证在sklearn中的模块为sklearn.model_selection
from sklearn.model_selection import cross_val_score
lr = LogisticRegression(C=100,max_iter=10000)   #最大迭代次数如果少了，可能对出现报错
scores = cross_val_score(lr, X_train, y_train, cv=10)
# array([0.85074627, 0.73134328, 0.8358209 , 0.76119403, 0.86567164,
#        0.86567164, 0.73134328, 0.82089552, 0.74242424, 0.74242424])
# 平均交叉验证分数
print("Average cross-validation score: {:.2f}".format(scores.mean()))
# Average cross-validation score: 0.79

'''任务二：混淆矩阵'''
# 计算二分类问题的混淆矩阵
# 计算精确率、召回率以及f-分数
# 混淆矩阵的方法在sklearn中的sklearn.metrics模块
# 混淆矩阵需要输入真实标签和预测标签
# 精确率、召回率以及f-分数可使用classification_report模块
from sklearn.metrics import confusion_matrix
lr = RandomForestClassifier()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
confusion_matrix(y_test, pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))


'''任务三：ROC曲线'''
# 绘制ROC曲线
# ROC曲线在sklearn中的模块为sklearn.metrics
# ROC曲线下面所包围的面积越大越好
lr = LogisticRegression(C=100,max_iter=10000)
lr.fit(X_train, y_train)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# 找到最接近于0的阈值
close_zero = np.argmin(np.abs(thresholds))  #argmin就是返回index
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold zero", 
         fillstyle="none", c='k', mew=2)
plt.legend(loc=4)










