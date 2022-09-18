# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:13:03 2022

@author: 18721
"""

# 导入基本库
import numpy as np
import pandas as pd
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
# 载入data文件中的:result.csv
text = pd.read_csv(path+'result.csv')
text.head()

# 2.6 数据运用
# 2.6.1 任务一：通过《Python for Data Analysis》P303、Google or Baidu来学习了解GroupBy机制
# 2.4.2：任务二：计算泰坦尼克号男性与女性的平均票价
text['Fare'].groupby(text['Sex']).mean()  #按照text['Sex']来对text['Fare']求取.mean()
# Sex
# female    44.479818
# male      25.523893
# Name: Fare, dtype: float64

# 2.4.3：任务三：统计泰坦尼克号中男女的存活人数
text['Survived'].groupby(text['Sex']).sum() #按照text['Sex']来对text['Survived']求取.sum() 
# Sex
# female    233
# male      109
# Name: Survived, dtype: int64

# 2.4.4：任务四：计算客舱不同等级的存活人数
text['Survived'].groupby(text['Pclass']).sum()
# Pclass
# 1    136
# 2     87
# 3    119
# Name: Survived, dtype: int64

# 【思考】从任务二到任务三中，这些运算可以通过agg()函数来同时计算。并且可以使用rename函数修改列名。
text.groupby('Sex').agg({'Fare': 'mean', 'Pclass': 'count'})
#              Fare  Pclass
# Sex                      
# female  44.479818     314
# male    25.523893     577

text.groupby('Sex').agg({'Fare': 'mean', 'Pclass': 'count'}).rename(columns=
                            {'Fare': 'mean_fare', 'Pclass': 'count_pclass'})
#         mean_fare  count_pclass  通过rename可以得到新的columns名字
# Sex                            
# female  44.479818           314
# male    25.523893           577

# 2.4.5：任务五：统计在不同等级的票中的不同年龄的船票花费的平均值
df1=text.groupby(['Pclass','Age'])['Fare'].mean()  #两级聚类，数据来源还是fare，但是聚类标准包括pclass和age
#这个聚类的结果比较多，是(182,)个
# Pclass  Age        部分结果为  
# 1       0.92     151.5500
#         2.00     151.5500
#         4.00      81.8583
#         11.00    120.0000
#         14.00    120.0000
  
# 3       61.00      6.2375
#         63.00      9.5875
#         65.00      7.7500
#         70.50      7.7500
#         74.00      7.7750

# 2.4.6：任务六：将任务二和任务三的数据合并，并保存到sex_fare_survived.csv
result2 = text['Fare'].groupby(text['Sex']).mean()
result3 =  text['Survived'].groupby(text['Sex']).sum()
#左右拼接，可以使用merge或者join，以及concat的axis=1实现
result6_1 = pd.merge(result2,result3,on='Sex')
#              Fare  Survived
# Sex                        
# female  44.479818       233
# male    25.523893       109
result6_2=pd.DataFrame(result2).join(pd.DataFrame(result3)) #注意：'Series' object has no attribute 'join'
#              Fare  Survived
# Sex                        
# female  44.479818       233
# male    25.523893       109
result6_3=pd.concat([result2,result3],axis=1)
#              Fare  Survived
# Sex                        
# female  44.479818       233
# male    25.523893       109

result6_2.equals(result6_1)
result6_2.equals(result6_3)


# 2.4.7：任务七：得出不同年龄的总的存活人数，然后找出存活人数最多的年龄段，
# 最后计算存活人数最高的存活率（存活人数/总人数）
#不同年龄的存活人数
survived_age = text['Survived'].groupby(text['Age']).sum()   #还有age=0.42的么
# text[text['Age']==0.42]

#不同年龄段存活人数最多的是多少？
survived_age.max() #15

#找出最大值的年龄段
survived_age[survived_age.values==survived_age.max()]
# Age
# 24.0    15    也就是24岁的存活人数最多，达到了15

#要计算存活人数最高的存活率
#首先计算总人数
_sum = text['Survived'].sum()
print("sum of person:"+str(_sum))
precetn =survived_age.max()/_sum
print("最大存活率："+str(precetn))
# sum of person:342   总共存活的人数
# 最大存活率：0.043859649122807015



















