# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:27:24 2022

@author: 18721
"""

# 在第二章第一节的内容中，我们学习了数据的清洗，这一部分十分重要，只有数据变得相对干净，
# 我们之后对数据的分析才可以更有力。而这一节，我们要做的是数据重构，数据重构依旧属于数据理解（准备）的范围。

# 导入基本库,建立数据集
import numpy as np
import pandas as pd
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
train=pd.read_csv(path+'train.csv')

train_left_up=train.iloc[:440,:4]
train_left_up.to_csv(path+'train_left_up.csv',index=False)

train_left_down=train.iloc[440:,:4]
train_left_down.to_csv(path+'train_left_down.csv',index=False)

train_right_up=train.iloc[:440,4:]
train_right_up.to_csv(path+'train_right_up.csv',index=False)

train_right_down=train.iloc[440:,4:]
train_right_down.to_csv(path+'train_right_down.csv',index=False)

# 2.4 数据的合并
# 2.4.1 任务一：将data文件夹里面的所有数据都载入，与之前的原始数据相比，观察他们的之间的关系
text_left_up = pd.read_csv(path+'train_left_up.csv')
text_left_down = pd.read_csv(path+'train_left_down.csv')
text_right_up = pd.read_csv(path+'train_right_up.csv')
text_right_down = pd.read_csv(path+'train_right_down.csv')

# 2.4.2：任务二：使用concat方法：将数据train-left-up.csv和train-right-up.csv横向合并为一张表，并保存这张表为result_up
list_up = [text_left_up,text_right_up]
result_up = pd.concat(list_up,axis=1)   #axis=1就是5*2 和5*3的axis=1的维度上的拼接
result_up.head()  #[5 rows x 12 columns]

# 2.4.3 任务三：使用concat方法：将train-left-down和train-right-down横向合并为一张表，并保存这张表为result_down。然后将上边的result_up和result_down纵向合并为result。
list_down=[text_left_down,text_right_down]
result_down = pd.concat(list_down,axis=1)
result = pd.concat([result_up,result_down])  #默认的axis=0
result.head()


#其实上面的concat方法比较好用
# 2.4.4 任务四：使用DataFrame自带的方法join方法和append：完成任务二和任务三的任务
# join其实是特征维度的操作，而append函数其实是样本维度的操作
resul_up = text_left_up.join(text_right_up)     #上面左右
result_down = text_left_down.join(text_right_down)   #下面左右
result = result_up.append(result_down)  #上下
result.head()

# 2.4.5 任务五：使用Panads的merge方法和DataFrame的append方法：完成任务二和任务三的任务
result_up = pd.merge(text_left_up,text_right_up,left_index=True,right_index=True)  #要按照共同的列进行merge
result_down = pd.merge(text_left_down,text_right_down,left_index=True,right_index=True)
result = result_up.append(result_down)  #append其实并不变
result.head()

# 2.4.6 任务六：完成的数据保存为result.csv
result.to_csv(path+'result.csv',index=False)
