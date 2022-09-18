# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:55:12 2022

@author: 18721
"""



# 数据清洗以及数据的特征处理，数据重构以及数据可视化。
#加载所需的库
import numpy as np
import pandas as pd
#加载数据train.csv
df = pd.read_csv('./data/train.csv')


'''我们拿到的数据通常是不干净的，所谓的不干净，就是数据中有缺失值，有一些异常点等，需要经过一定的处理才能继续做后面的分析或建模，
缺失值其实很好处理，异常值好像是可以使用孤立森林来处理

所以拿到数据的第一步是进行数据清洗，本章我们将学习缺失值、重复值、字符串和数据转换等操作，将数据清洗成可以分析或建模的样子。
缺失值直接填充，重复值怎么办？字符串可能是分类数据，转换成类别数据即可'''

# 2.1 缺失值观察与处理
# 2.1.1 任务一：缺失值观察
# (1) 请查看每个特征缺失值个数
# (2) 请查看Age， Cabin， Embarked列的数据
# (1)
df.info()
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64  
#  2   Pclass       891 non-null    int64  
#  3   Name         891 non-null    object 
#  4   Sex          891 non-null    object 
#  5   Age          714 non-null    float64     非null数目不等于891的都是存在缺失值的
#  6   SibSp        891 non-null    int64  
#  7   Parch        891 non-null    int64  
#  8   Ticket       891 non-null    object 
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object 
#  11  Embarked     889 non-null    object 
#如果只是想要看缺失值的个数，可以先isna()或者isnull(),然后用sum对true进行求和
df.isnull().sum()  
df.isna().sum()
# (2)
df[['Age','Cabin','Embarked']]  #[891 rows x 3 columns]


# 2.1.2 任务二：对缺失值进行处理
# (1)处理缺失值一般有几种思路
#可以直接删除，pandas课程的7.1.2节就是使用dropna函数进行删除
#可以填充，pandas课程的7.2.1节的fillna函数，还有插值函数，其实使用的不是很多，还是fillna用的更多


# (2) 请尝试对Age列的数据的缺失值进行处理
#方式一，赋值
#例如最简单的就是，只要是缺失值我就直接设置为0
df[df['Age']==None]=0
df['Age'].info()  #714 non-null 可见其实并没有实现nan赋值为0
(df['Age']==None).sum()  #输出0，表明nan和这none并不是相等的

df[df['Age'].isnull()] = 0 # 还好
df['Age'].info()   #891 non-null 可见使用isnull函数确实能够识别出来nan

df[df['Age'] == np.nan] = 0  
df['Age'].info()   #714 non-null,也不行
(df['Age'] == np.nan).sum() #输出的也是0

# (3) 请尝试使用不同的方法直接对整张表的缺失值进行处理
#方式二，删除
df.dropna()  #那么现在其实只有183列了
df.dropna().info()  #每一列都没有缺失值，说明how函数使用的是any
#方式三，填充
df.fillna(0).info()
df.info()  #只是副本上填充









