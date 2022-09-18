# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:31:51 2022

@author: 18721
"""

# 我们拥有的泰坦尼克号的数据集，那么我们这次的目的就是，完成泰坦尼克号存活预测这个任务。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
# 读取原数据数集
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
train = pd.read_csv(path+'train.csv')
train.shape  #(891, 12)

# train['Sex'].unique()   可以整数编码或者独热编码
# train['Sex'].value_counts()
# array(['male', 'female'], dtype=object)
# male      577
# female    314
# Name: Sex, dtype: int64

train['Embarked'].value_counts()
train['Embarked'].unique()
# S    644
# C    168
# Q     77
# Name: Embarked, dtype: int64
# array(['S', 'C', 'Q', nan], dtype=object)

train.info()
 # 0   PassengerId  891 non-null    int64  
 # 1   Survived     891 non-null    int64  
 # 2   Pclass       891 non-null    int64  
 # 3   Name         891 non-null    object 
 # 4   Sex          891 non-null    object 
 # 5   Age          714 non-null    float64
 # 6   SibSp        891 non-null    int64  
 # 7   Parch        891 non-null    int64  
 # 8   Ticket       891 non-null    object 
 # 9   Fare         891 non-null    float64
 # 10  Cabin        204 non-null    object 
 # 11  Embarked     889 non-null    object 
 
train_copy=train.copy()
def sex(x):
    if x=='female':
        return 0
    elif x=='male':
        return 1
    else:
        return None
train_copy['Sex']=train_copy['Sex'].apply(sex)
def embarked(x):
    if x=='S':
        return 0
    elif x=='C':
        return 1
    elif x=='Q':
        return 2
    else:
        return None
train_copy['Embarked']=train_copy['Embarked'].apply(embarked)
# corr_matrix=train_copy.corr()
# corr_matrix['Survived'].sort_values(ascending=False)
# Survived       1.000000
# Fare           0.257307
# Parch          0.081629   没有什么相关性
# PassengerId   -0.005007   没有什么相关性
# SibSp         -0.035322   没有什么相关性
# Age           -0.077221   没有什么相关性，但是很奇怪，为什么跟年龄没有相关性呢，也许把年龄和性别挂钩之后可能就存在相关性了
# Pclass        -0.338481
# Sex           -0.543351

#读取清洗过的数据集
data = pd.read_csv(path+'clean_data.csv')
# data['Survived']=train['Survived']
# corr_matrix=data.corr()
# corr_matrix['Survived'].sort_values(ascending=False)
# Survived       1.000000
# Sex_female     0.543351     女性和存活有着正相关
# Fare           0.257307
# Embarked_C     0.168240
# Parch          0.081629
# Age            0.010539
# Embarked_Q     0.003650
# PassengerId   -0.005007
# SibSp         -0.035322
# Embarked_S    -0.155660
# Pclass        -0.338481
# Sex_male      -0.543351     男性和存活有着负相关

# 任务一：切割训练集和测试集
# 这里使用留出法划分数据集   没有使用K折交叉验证
# 将数据集分为自变量和因变量
# 按比例切割训练集和测试集(一般测试集的比例有30%、25%、20%、15%和10%)
# 使用分层抽样
# 设置随机种子以便结果能复现
from sklearn.model_selection import train_test_split
# 一般先取出X和y后再切割，有些情况会使用到未切割的，这时候X和y就可以用,x是清洗好的数据，y是我们要预测的存活数据'Survived'
X = data
y = train['Survived']
# 对数据集进行切割，而且按照y进行分层抽样，使得分类问题是比较平衡的，而且设置了随机数种子random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# 查看数据形状
X_train.shape, X_test.shape  # ((668, 11), (223, 11))


# 任务二：模型创建
# 创建基于线性模型的分类模型（逻辑回归）
# 创建基于树的分类模型（决策树、随机森林）
# 分别使用这些模型进行训练，分别的到训练集和测试集的得分
# 查看模型的参数，并更改参数值，观察模型变化
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# model1：默认参数逻辑回归模型
lr = LogisticRegression()
lr.fit(X_train, y_train)
# 查看训练集和测试集score值，问？这个score是什么，好像就是R2
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(lr.score(X_test, y_test)))
# Training set score: 0.80
# Testing set score: 0.77


#model2 调整参数后的逻辑回归模型
lr2 = LogisticRegression(C=100)   #调整了参数C，C越小，lambda越大，则正则化强度越大，本来默认的是1
lr2.fit(X_train, y_train)
print("Training set score: {:.2f}".format(lr2.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(lr2.score(X_test, y_test)))
# Training set score: 0.80
# Testing set score: 0.78  

# model3：默认参数的随机森林分类模型
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print("Training set score: {:.2f}".format(rfc.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(rfc.score(X_test, y_test)))
# Training set score: 1.00   也就是完全满足
# Testing set score: 0.82

# model4：调整参数后的随机森林分类模型，限制最大深度其实就会让模型正则化
rfc2 = RandomForestClassifier(n_estimators=100, max_depth=5)  
rfc2.fit(X_train, y_train)
print("Training set score: {:.2f}".format(rfc2.score(X_train, y_train)))
print("Testing set score: {:.2f}".format(rfc2.score(X_test, y_test)))
# Training set score: 0.86   好像并不好，这个时候其实就很需要使用网格搜索等等实现
# Testing set score: 0.80

# 任务三：输出模型预测结果
# 输出模型预测分类标签
# 输出不同分类标签的预测概率
# 一般监督模型在sklearn里面有个predict能输出预测标签，predict_proba则可以输出标签概率
# 预测标签
pred = lr.predict(X_train)
pred_proba = lr.predict_proba(X_train)
from sklearn.metrics import accuracy_score as acc
acc(pred,y_train)  #0.7964071856287425
acc(lr.predict(X_test),y_test) # 0.7713004484304933

