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

# 2.2 重复值观察与处理
# 2.2.1 任务一：请查看数据中的重复值
df = pd.read_csv('./data/train.csv')
df[df.duplicated()] 
# 2.2.2 任务二：对重复值进行处理
# (1)重复值有哪些处理方式呢？
# (2)处理我们数据的重复值
df = df.drop_duplicates()   #删掉重复值
#我们创造一个有重复值的dataframe
df1=pd.DataFrame([np.arange(1,11) for i in range(4)],
                 columns=list('abcdefghij'),
                 index=list('ABCD'))  #创建4行10列，每一行都是1-10的数据
df1 = df1.drop_duplicates()  #会变成一行，也就是只保留一个sample的样本
# 2.2.3 任务三：将前面清洗的数据保存为csv格式
df.to_csv('test_clear.csv')

# 2.3 特征观察与处理
df = pd.read_csv('./data/train.csv')
df.info()
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
'''我们对特征进行一下观察，可以把特征大概分为两大类：
数值型特征：Survived ，Pclass， Age ，SibSp， Parch， Fare，其中Survived， Pclass为离散型数值特征，Age，SibSp， Parch， Fare为连续型数值特征
文本型特征：Name， Sex， Cabin，Embarked， Ticket，其中Sex， Cabin， Embarked， Ticket为类别型文本特征。'''
'''对于上面两种特征：
数值型特征一般可以直接用于模型的训练，但有时候为了模型的稳定性及鲁棒性会对连续变量进行离散化。
文本型特征往往需要转换成数值型特征才能用于建模分析。'''

# 2.3.1 任务一：对年龄进行分箱（离散化）处理，年龄本来是连续的数值特征，为了模型的鲁棒性，我们离散化处理
# (1) 分箱操作是什么？    就是分成区间
# (2) 将连续变量Age平均分箱成5个年龄段，并分别用类别变量12345表示
#将连续变量Age平均分箱成5个年龄段，并分别用类别变量12345表示
df['AgeBand'] = pd.cut(df['Age'], 5,labels = [1,2,3,4,5])
df['AgeBand'].unique()
# [2, 3, NaN, 4, 1, 5]  
# Categories (5, int64): [1 < 2 < 3 < 4 < 5]  有序的分类
for i in range(1,6):
    print((df['AgeBand']==i).sum())   
# 100   可见每个区间的数目并不是相等的
# 346
# 188
# 69
# 11
for i in range(1,6):
    print('区间标签为：',str(i),'，此区间的最大值为：',(df['Age'][df['AgeBand']==i]).max(),'，此区间的最小值为：'\
          ,(df['Age'][df['AgeBand']==i]).min())  
# 区间标签为： 1 ，此区间的最大值为： 16.0 ，此区间的最小值为： 0.42  可见区间是等间隔划分的
# 区间标签为： 2 ，此区间的最大值为： 32.0 ，此区间的最小值为： 17.0
# 区间标签为： 3 ，此区间的最大值为： 48.0 ，此区间的最小值为： 32.5
# 区间标签为： 4 ，此区间的最大值为： 64.0 ，此区间的最小值为： 49.0
# 区间标签为： 5 ，此区间的最大值为： 80.0 ，此区间的最小值为： 65.0

# (3) 将连续变量Age划分为(0,5] (5,15] (15,30] (30,50] (50,80]五个年龄段，并分别用类别变量12345表示
#将连续变量Age划分为(0,5] (5,15] (15,30] (30,50] (50,80]五个年龄段，并分别用类别变量12345表示
df['AgeBand1'] = pd.cut(df['Age'],[0,5,15,30,50,80],labels = [1,2,3,4,5])
for i in range(1,6):
    print((df['AgeBand1']==i).sum())  
# 44
# 39
# 326
# 241
# 64
for i in range(1,6):
    print('区间标签为：',str(i),'，此区间的最大值为：',(df['Age'][df['AgeBand1']==i]).max(),'，此区间的最小值为：'\
          ,(df['Age'][df['AgeBand1']==i]).min()) 
# 区间标签为： 1 ，此区间的最大值为： 5.0 ，此区间的最小值为： 0.42   可见区间的最大值就是我们自己设置的
# 区间标签为： 2 ，此区间的最大值为： 15.0 ，此区间的最小值为： 6.0
# 区间标签为： 3 ，此区间的最大值为： 30.0 ，此区间的最小值为： 16.0
# 区间标签为： 4 ，此区间的最大值为： 50.0 ，此区间的最小值为： 30.5
# 区间标签为： 5 ，此区间的最大值为： 80.0 ，此区间的最小值为： 51.0

# (4) 将连续变量Age按10% 30% 50% 70% 90%五个年龄段，并用分类变量12345表示
#将连续变量Age按10% 30% 50 70% 90%五个年龄段，并用分类变量12345表示，下面的qcut中的q其实就是quantile的意思，分位数的意思
df['AgeBand2'] = pd.qcut(df['Age'],[0,0.1,0.3,0.5,0.7,0.9],labels = [1,2,3,4,5])
count=[]
for i in range(1,6):
    print((df['AgeBand2']==i).sum())  
    count.append((df['AgeBand2']==i).sum())
count=np.array(count)
cum_count=np.cumsum(count)   
cum_count=cum_count/np.max(cum_count)
# array([0.11846154, 0.35538462, 0.55692308, 0.79846154, 1.        ]) 这好像不对，接下来再去看看pandas中的课程
# (5) 将上面的获得的数据分别进行保存，保存为csv格式


# 2.3.2 任务二：对文本变量进行转换
# (1) 查看文本变量名及种类  Sex， Cabin， Embarked， Ticket为类别型文本特征
#第一种，value_counts可以返回每一种的名字及其数量，这一种方式最为常见
df['Sex'].value_counts()
# male      577
# female    314
df['Cabin'].value_counts()   #有很多种
df['Embarked'].value_counts()
# S    644
# C    168
# Q     77
#第二种，使用unique可以得到每一种的名字
df['Sex'].unique()  # array(['male', 'female'], dtype=object)
df['Cabin'].unique()  #一个长度为148的array
df['Embarked'].unique() # array(['S', 'C', 'Q', nan], dtype=object)
#第三种，使用.nunique()可以得到不同类别的数目
df['Sex'].nunique()  # 2
df['Cabin'].nunique()  #147,不包括nan
df['Embarked'].nunique() # 3 ,不包括nan

# (2) 将文本变量Sex， Cabin ，Embarked用数值变量12345表示，这其实就是整数编码
#第一种，使用的是replace函数
df['Sex_num'] = df['Sex'].replace(['male','female'],[1,2])
#第二种，map后面加上一个字典
df['Sex_num1'] = df['Sex'].map({'male': 1, 'female': 2})
#问？上面两种是否相等呢？
df['Sex_num'].equals(df['Sex_num1'])  #确实输出true
#方法三: 使用sklearn.preprocessing的LabelEncoder,这一块其实可以看一下hands书的整数编码和独热编码两种方式
from sklearn.preprocessing import LabelEncoder
for feat in ['Cabin', 'Ticket']:  #哪一列需要处理
    # feat='Embarked'
    lbl = LabelEncoder()    #实例化
    label_dict = dict(zip(df[feat].unique(), range(df[feat].nunique())))  #将每种类别映射到0-n的整数  {'S': 0, 'C': 1, 'Q': 2}
    df[feat + "_labelEncode"] = df[feat].map(label_dict)
    # df.info()  Embarked_labelEncode  889 non-null    float64 
    df[feat + "_labelEncode"] = lbl.fit_transform(df[feat].astype(str))
    df.info()

# (3) 将文本变量Sex， Cabin， Embarked用one-hot编码表示
#方法一: OneHotEncoder
for feat in ["Age", "Embarked"]:
    # feat="Embarked"
    x = pd.get_dummies(df[feat], prefix=feat)  #就得到了891*3的数据
    df = pd.concat([df, x], axis=1)  #然后拼接即可，axis=1指的是891*2+891*3 ,在2和3的位置上拼接
    

# 2.3.3 任务三（附加）：从纯文本Name特征里提取出Titles的特征(所谓的Titles就是Mr,Miss,Mrs等)
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)  #这就是正则表达式，关键是.  这个点能够让.之前的非空格字符被提取出来








