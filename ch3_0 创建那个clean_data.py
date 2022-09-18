# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:56:10 2022

@author: 18721
"""
import pandas as pd
import numpy as np
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
#我们先创建clean_data
clean_data=pd.read_csv(path+'train.csv',
                       usecols=['PassengerId','Pclass','Age','SibSp','Parch','Fare','Sex','Embarked'])
#对sex和embarked进行独热编码
for feat in ["Sex", "Embarked"]:
    x = pd.get_dummies(clean_data[feat], prefix=feat)  #就得到了891*3的数据
    clean_data = pd.concat([clean_data, x], axis=1)
clean_data=clean_data.drop(['Sex','Embarked'],axis=1)

clean_data.info()
 # 0   PassengerId  891 non-null    int64  
 # 1   Pclass       891 non-null    int64  
 # 2   Age          714 non-null    float64    这一列存在nan
 # 3   SibSp        891 non-null    int64  
 # 4   Parch        891 non-null    int64  
 # 5   Fare         891 non-null    float64
 # 6   Sex_female   891 non-null    uint8  
 # 7   Sex_male     891 non-null    uint8  
 # 8   Embarked_C   891 non-null    uint8  
 # 9   Embarked_Q   891 non-null    uint8  
 # 10  Embarked_S   891 non-null    uint8 
clean_data=clean_data.fillna(0)  #填充了很多的0
clean_data.head()
clean_data.to_csv(path+'clean_data.csv',index=False)
