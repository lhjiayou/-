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









