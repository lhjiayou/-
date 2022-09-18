# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:55:54 2022

@author: 18721
"""

import numpy as np
import pandas as pd
path=r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/'
# train=pd.read_csv(path+'train.csv')

# 2.5 换一种角度看数据
# 2.5.1 任务一：将我们的数据变为Series类型的数据
# 将完整的数据加载出来
text = pd.read_csv(path+'result.csv')
text.head()
text.shape  #(891, 12)
# 代码写在这里
unit_result=text.stack()  #一条一条堆叠起来
unit_result.head()
#将代码保存为unit_result,csv
unit_result.to_csv(path+'unit_result.csv') #不要证书索引index=False和表头header=False
test = pd.read_csv(path+'unit_result.csv')
