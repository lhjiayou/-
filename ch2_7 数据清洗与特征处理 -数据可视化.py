# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:14:33 2022

@author: 18721
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
text = pd.read_csv(path+'result.csv')

#2.7 数据可视化
# 2.7.1 任务一：跟着书本第九章，了解matplotlib，自己创建一个数据项，对其进行基本可视化
# 【思考】最基本的可视化图案有哪些？分别适用于那些场景？（比如折线图适合可视化某个属性值随时间变化的走势）


# 2.7.2 任务二：可视化展示泰坦尼克号数据集中男女中生存人数分布情况（用柱状图试试）。
sex = text.groupby('Sex')['Survived'].sum()    #按照sex进行聚合，数据来源是survived，操作是求和
# Sex
# female    233
# male      109
sex.plot.bar()      #对series直接调用这个方法么？
# plt.bar(sex.index,sex.values)
plt.title('survived_count')
plt.show()

# 2.7.3 任务三：可视化展示泰坦尼克号数据集中男女中生存人与死亡人数的比例图（用柱状图试试）。
# 提示：计算男女中死亡人数 1表示生存，0表示死亡
text.groupby(['Sex','Survived'])['Survived'].count() #表示按照sex和survived分组进行统计
# Sex     Survived
# female  0            81
#         1           233
# male    0           468
#         1           109
text.groupby(['Sex','Survived'])['Survived'].count().unstack()
#将上面的结果的内层index转换到列中，就是unstack的作用
# Survived    0    1
# Sex               
# female     81  233   可见女性存活更多
# male      468  109
text.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
#画图，设置类别为bar，而且是堆叠图
plt.title('survived_count')
plt.ylabel('count')

# 2.7.4 任务四：可视化展示泰坦尼克号数据集中不同票价的人生存和死亡人数分布情况。（用折线图试试）（横轴是不同票价，纵轴是存活人数）
# 计算不同票价中生存与死亡人数 1表示生存，0表示死亡
# fare_sur = text.groupby(['Fare'])['Survived'].count()  

#降序排列后的折线图
text.groupby(['Fare'])['Survived'].value_counts()
fare_sur=text.groupby(['Fare'])['Survived'].value_counts().sort_values(ascending=False) #降序排列统计值
fig = plt.figure(figsize=(20, 18))
fare_sur.plot(grid=True)
plt.legend()
plt.show()

# 排序前绘折线图
fare_sur1 = text.groupby(['Fare'])['Survived'].value_counts()
fig = plt.figure(figsize=(20, 18))
fare_sur1.plot(grid=True)
plt.legend()
plt.show()

# 2.7.5 任务五：可视化展示泰坦尼克号数据集中不同仓位等级的人生存和死亡人员的分布情况。（用柱状图试试）
# 1表示生存，0表示死亡
pclass_sur = text.groupby(['Pclass'])['Survived'].value_counts()
import seaborn as sns
sns.countplot(x="Pclass", hue="Survived", data=text)
# sns.countplot() 用于画类别特征的频数条形图。
# hue：在x或y标签划分的同时，再以hue标签划分统计个数
#加上了hue之后，可见数据就是不同的x的位置有两列
# data：df或array或array列表，用于绘图的数据集，x或y缺失时，data参数为数据集，同时x或y不可缺少，必须要有其中一个

# 2.7.6 任务六：可视化展示泰坦尼克号数据集中不同年龄的人生存与死亡人数分布情况。(不限表达方式)
facet = sns.FacetGrid(text, hue="Survived",aspect=3)  #实例化一个库，aspect是纵横比的意思
facet.map(sns.kdeplot,'Age',shade= True)   #kde就是核密度估计的意思，调用map函数
facet.set(xlim=(0, text['Age'].max()))  #设置x轴的范围
facet.add_legend()  #添加图例
#如果要画成2.7.7中的折线图
text.Age[text.Survived == 0].plot(kind='kde',label='0')
text.Age[text.Survived == 1].plot(kind='kde',label='1')
plt.xlabel("age")
plt.legend(loc="best")

# 2.7.7 任务七：可视化展示泰坦尼克号数据集中不同仓位等级的人年龄分布情况。（用折线图试试）
#不同于上面的sns的kde核密度估计图，这个是折线图
text.Age[text.Pclass == 1].plot(kind='kde')
text.Age[text.Pclass == 2].plot(kind='kde')
text.Age[text.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.legend((1,2,3),loc="best")

































