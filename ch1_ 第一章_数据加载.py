# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:45:33 2022

@author: 18721
"""

# 1 第一章：数据加载

'''1.1 载入数据'''

# 1.1.1 任务一：导入numpy和pandas
import numpy as np
import pandas as pd

# 1.1.2 任务二：载入数据
# (1) 使用相对路径载入数据
df_train=pd.read_csv('./data/train.csv')
df_train.shape  # (891, 12)
df_test=pd.read_csv('./data/test.csv')
df_test.shape  #(418, 11)
# (2) 使用绝对路径载入数据
df_train_absolute=pd.read_csv(r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/train.csv')
df_train_absolute.shape  # (891, 12)
df_test_absolute=pd.read_csv(r'D:\！datawhale学习\动手学数据分析\hands-on-data-analysis-master\data/test.csv')
df_test_absolute.shape  #(418, 11)
#验证一下加载的数据完全相同,输出的确实是true
df_train_absolute.equals(df_train)
df_test_absolute.equals(df_test)
#虽然pd.read_csv()和pd.read_table()有所不同，但是目前看来大部分竞赛数据都是使用pd.read_csv()读取的
# pd.read_csv() 从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为逗号  ，也就是读取excel生成的csv文件
# pd.read_table() 从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为制表符“\t”，也就是读取txt文件

# 1.1.3 任务三：每1000行为一个数据模块，逐块读取
train_path='./data/train.csv'
test_path='./data/test.csv'
chunker = pd.read_csv(train_path, chunksize=1000)
# 什么是逐块读取？为什么要逐块读取呢？
# 处理大文件时，只需要读取文件的一小部分或逐块对文件进行迭代。当数据集太大时，通过逐块读取数据可以加快文件读取的速度。
for chunker in df_train:
    print(chunker)
    
#1.1.4 任务四：将表头改成中文（直接赋值即可），索引改为乘客ID [对于某些英文资料，我们可以通过翻译来更直观的熟悉我们的数据]
df_train_chinese = pd.read_csv(train_path, 
                              names=['乘客ID','是否幸存','仓位等级','姓名','性别','年龄','兄弟姐妹个数',
                                       '父母子女个数','船票信息','票价','客舱','登船港口'],
                              #上面的names其实就是读入的时候给每一列设置的column_name
                              index_col='乘客ID',
                              #将乘客ID设置为index，则整个数据变成了11列
                              header=0 #明确设定header=0就会替换掉原来存在列名
                              )
df_train_chinese.head()
#总而言之就是之前的df_train本来是891*12，现在的df_train_chinese是891*11
#原来的英文列名全部设置成了现在的中文列名，而且将乘客ID设置为index，舍弃了之前的默认自然整数index


'''1.2 初步观察'''
# 导入数据后，你可能要对数据的整体结构和样例进行概览，比如说，数据大小、有多少列，各列都是什么格式的，是否包含null等

# 1.2.1 任务一：查看数据的基本信息
df_train_chinese.info()
# Int64Index: 891 entries, 1 to 891
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   是否幸存    891 non-null    int64  
#  1   仓位等级    891 non-null    int64  
#  2   姓名      891 non-null    object 
#  3   性别      891 non-null    object 
#  4   年龄      714 non-null    float64     小于891的都是存在缺失值的
#  5   兄弟姐妹个数  891 non-null    int64  
#  6   父母子女个数  891 non-null    int64  
#  7   船票信息    891 non-null    object 
#  8   票价      891 non-null    float64
#  9   客舱      204 non-null    object 
#  10  登船港口    889 non-null    object 


# 1.2.2 任务二：观察表格前10行的数据和后15行的数据
df_train_chinese.head(10)
df_train_chinese.tail(15)


# 1.2.3 任务三：判断数据是否为空，为空的地方返回True，其余地方返回False,可以使用isnull或者isna函数
df_train_chinese.isnull().head()
df_train_chinese.isna().head()

'''1.3 保存数据'''
# 1.3.1 任务一：将你加载并做出改变的数据，在工作目录下保存为一个新文件train_chinese.csv
df_train_chinese.to_csv('./data/train_chinese.csv')  #确实出现了中文字符的乱码现象
df_train_chinese.to_csv('./data/train_chinese.csv',encoding='utf-8')  #仍然乱码
df_train_chinese.to_csv('./data/train_chinese.csv',encoding='GBK')   #GBK才是符合中国标准的国标

'''1.4 知道你的数据叫什么'''
# 1.4.1 任务一：pandas中有两个数据类型DateFrame和Series，通过查找简单了解他们。然后自己写一个关于这两个数据类型的小例子?[开放题]
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
example_1 = pd.Series(sdata)  #这其实就是通过字典创建series
example_1
example_2=pd.Series(example_1.values)  #这其实是通过array创建series，但是其index是默认的整数索引
example_2.index=example_1.index
example_2

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
example_2 = pd.DataFrame(data)  #仍然是通过字典创建dataframe，其column就是字典的key
example_2

#1.4.2 任务二：根据上节课的方法载入"train.csv"文件
df_train_chinese=pd.read_csv('./data/train_chinese.csv',encoding='GBK')  #自动加上了默认的整数索引
df_train_chinese=pd.read_csv('./data/train_chinese.csv',encoding='GBK').set_index(['乘客ID'])   #不需要默认的整数index，而是将乘客ID作为index


# 1.4.3 任务三：查看DataFrame数据的每列的名称
df_train_chinese.columns  #1.4.2节的两种方式会使得乘客ID作为列存在差异

# 1.4.4任务四：查看"Cabin"这列的所有值 [有多种方法]
df_train=pd.read_csv('./data/train.csv') 
df_train.columns
#获取一列，可以使用['']或者句点表示法
df_train['Cabin']
df_train.Cabin


# 1.4.5 任务五：加载文件"test_1.csv"，然后对比"train.csv"，看看有哪些多出的列，然后将多出的列删除
#直接从kaggle上下载的其实没有这个文件
df_test=pd.read_csv('./data/test.csv') 
df_test_1=df_test.copy()
df_test_1['result']=np.arange(418)  #随便创造了一列
del df_test_1['result']  #成功删除一列数据
#也可以直接选择dataframe的subset的方式，丢弃那一列
df_test_1['result']=np.arange(418)
df_test_1=df_test_1.iloc[:,:-1]  #当然如果这一列不是最后一列，其实还是del更加好用


# 1.4.6 任务六： 将['PassengerId','Name','Age','Ticket']这几个列元素隐藏，只观察其他几个列元素
df_train.drop(['PassengerId','Name','Age','Ticket'],axis=1).head(3)  
#本来12列，现在还剩8列，但是注意只是在副本上删除的，原来的df_train并没有发生改变，除非将结果重新赋值给df_train，或者drop的时候inplace=True
#如果想要完全的删除你的数据结构，使用inplace=True，因为使用inplace就将原数据覆盖了，所以这里没有用
#方式一，重新赋值
df_train=pd.read_csv('./data/train.csv') 
df_train=df_train.drop(['PassengerId','Name','Age','Ticket'],axis=1)
#方式二，inplace=True
df_train=pd.read_csv('./data/train.csv') 
df_train.drop(['PassengerId','Name','Age','Ticket'],axis=1,inplace=True)    


'''1.5 筛选的逻辑'''
# 1.5.1 任务一： 我们以"Age"为筛选条件，显示年龄在10岁以下的乘客信息。
df_train=pd.read_csv('./data/train.csv') 
df_train[df_train["Age"]<10]  #[62 rows x 12 columns]可以筛选出62条记录
# 1.5.2 任务二： 以"Age"为条件，将年龄在10岁以上和50岁以下的乘客信息显示出来，并将这个数据命名为midage
midage = df_train[(df_train["Age"]>10)& (df_train["Age"]<50)]  #[576 rows x 12 columns]，此时的index不是连续的
# 1.5.3 任务三：将midage的数据中第100行的"Pclass"和"Sex"的数据显示出来，那么就需要将上面不连续的index重置
midage = midage.reset_index(drop=True) #重置index，而且将之前的index给丢弃
midage.loc[99,["Pclass",'Name',"Sex"]]  #第100行的index是99
# Pclass                                    2
# Name      Byles, Rev. Thomas Roussel Davids
# Sex                                    male
# Name: 100, dtype: object

midage.loc[[100],['Pclass','Name','Sex']] #不使用reset_index的话，这个就是index=100的，而不是第100行的数据
midage.iloc[100][['Pclass','Name','Sex']] #不使用reset_index，但是我们可以使用iloc来定位索引
# Pclass                                    2
# Name      Byles, Rev. Thomas Roussel Davids
# Sex                                    male
# Name: 149, dtype: object

# 1.5.4 任务四：使用loc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来
midage = df_train[(df_train["Age"]>10)& (df_train["Age"]<50)]
midage = midage.reset_index(drop=True)
midage.loc[[100,105,108],['Pclass','Name','Sex']]
#      Pclass                               Name   Sex
# 100       2  Byles, Rev. Thomas Roussel Davids  male
# 105       3           Cribb, Mr. John Hatfield  male
# 108       3                    Calic, Mr. Jovo  male

# 1.5.5 任务五：使用iloc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来
midage = df_train[(df_train["Age"]>10)& (df_train["Age"]<50)]
#如果不reset_index：
midage.iloc[[100,105,108],[2,3,4]]
#      Pclass                               Name   Sex
# 149       2  Byles, Rev. Thomas Roussel Davids  male
# 160       3           Cribb, Mr. John Hatfield  male
# 163       3                    Calic, Mr. Jovo  male
#如果reset_index：
midage = midage.reset_index(drop=True)
midage.iloc[[100,105,108],[2,3,4]]
#      Pclass                               Name   Sex
# 100       2  Byles, Rev. Thomas Roussel Davids  male
# 105       3           Cribb, Mr. John Hatfield  male
# 108       3                    Calic, Mr. Jovo  male


# 探索性数据分析 EDA
#载入之前保存的train_chinese.csv数据，关于泰坦尼克号的任务，我们就使用这个数据
text = pd.read_csv('./data/train_chinese.csv',encoding='GBK')
text.head()

'''1.6 了解你的数据吗？'''
# 1.6.1 任务一：利用Pandas对示例数据进行排序，要求升序
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), 
                     index=['2', '1'], 
                     columns=['d', 'a', 'b', 'c'])
frame
#    d  a  b  c
# 2  0  1  2  3
# 1  4  5  6  7
# 大多数时候我们都是想根据列的值来排序,所以，将你构建的DataFrame中的数据根据某一列，升序排列
frame.sort_values(by='c', ascending=False)   #降序排列，但是其实并没有改变本来的frame
#by参数指向要排列的列，按照某列进行排序，ascending参数指向排序的方式（升序还是降序）
#    d  a  b  c
# 1  4  5  6  7
# 2  0  1  2  3
# 让行索引升序排序
frame.sort_index()  #因为默认的axis=0，也就是对index进行排序
#    d  a  b  c
# 1  4  5  6  7
# 2  0  1  2  3
# 让列索引升序排序
frame.sort_index(axis=1)  #axis=1，指的是对column进行排序
#    a  b  c  d
# 2  1  2  3  0
# 1  5  6  7  4
# 让列索引降序排序
frame.sort_index(axis=1, ascending=False)  #设置ascending=False即可
#    d  c  b  a
# 2  0  3  2  1
# 1  4  7  6  5

# 1.6.2 任务二：对泰坦尼克号数据（trian.csv）按票价和年龄两列进行综合排序（降序排列），从数据中你能发现什么
result=text.sort_values(by=['票价', '年龄'], ascending=False)   
#也就是先按照票价进行排序，票价相等的时候按照年龄进行排序，如果票价和年龄都相等，就按照index进行排序
'''那么我们后面是不是可以进一步分析一下票价和存活之间的关系，年龄和存活之间的关系呢？当你开始发现数据之间的关系了，数据分析就开始了。'''
text['票价'].corr(text['是否幸存'])  #0.2573065223849622，相关系数
text['年龄'].corr(text['是否幸存']) #-0.07722109457217756 轻微负相关，也就是年龄越大，幸存概率越低
#好像可以计算一个相关矩阵，这一块我们按照机器学习实战2.4.2节的寻找相关性进行探索
corr_matrix=text.corr()
#然后按照是否幸存列的相关性进行排列
corr_matrix['是否幸存'].sort_values(ascending=False)
# 是否幸存      1.000000
# 票价        0.257307      #明显正相关
# 父母子女个数    0.081629
# 乘客ID     -0.005007
# 兄弟姐妹个数   -0.035322
# 年龄       -0.077221
# 仓位等级     -0.338481    #明显负相关
# Name: 是否幸存, dtype: float64

# 1.6.3 任务三：利用Pandas进行算术计算，计算两个DataFrame数据相加结果
frame1_a = pd.DataFrame(np.arange(9.).reshape(3, 3),
                     columns=['a', 'b', 'c'],
                     index=['one', 'two', 'three'])
#          a    b    c
# one    0.0  1.0  2.0
# two    3.0  4.0  5.0
# three  6.0  7.0  8.0
frame1_b = pd.DataFrame(np.arange(12.).reshape(4, 3),
                     columns=['a', 'e', 'c'],
                     index=['first', 'one', 'two', 'second'])
#           a     e     c
# first   0.0   1.0   2.0
# one     3.0   4.0   5.0
# two     6.0   7.0   8.0
# second  9.0  10.0  11.0
#将frame_a和frame_b进行相加
frame1_a + frame1_b
#           a   b     c   e
# first   NaN NaN   NaN NaN
# one     3.0 NaN   7.0 NaN   也就是只有共同的行和共同的列才能进行相加
# second  NaN NaN   NaN NaN
# three   NaN NaN   NaN NaN
# two     9.0 NaN  13.0 NaN

# 1.6.4 任务四：通过泰坦尼克号数据如何计算出在船上最大的家族有多少人？
max(text['兄弟姐妹个数'] + text['父母子女个数'])  #输出10

# 1.6.5 任务五：学会使用Pandas describe()函数查看数据基本统计信息
'''info反映是否存在缺失值，而describe反映分位数情况'''
frame2 = pd.DataFrame([[1.4, np.nan], 
                       [7.1, -4.5],
                       [np.nan, np.nan], 
                       [0.75, -1.3]
                      ], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
#     one  two
# a  1.40  NaN
# b  7.10 -4.5
# c   NaN  NaN
# d  0.75 -1.3
# 调用 describe 函数，观察frame2的数据基本信息，这是各种统计指标
frame2.describe()
#             one       two
# count  3.000000  2.000000
# mean   3.083333 -2.900000
# std    3.493685  2.262742
# min    0.750000 -4.500000
# 25%    1.075000 -3.700000
# 50%    1.400000 -2.900000
# 75%    4.250000 -2.100000
# max    7.100000 -1.300000

# 1.6.6 任务六：分别看看泰坦尼克号数据集中 票价、父母子女 这列数据的基本统计数据，你能发现什么？
text['票价'].describe()
# count    891.000000   表明不存在缺失值
# mean      32.204208
# std       49.693429   波动较大
# min        0.000000
# 25%        7.910400
# 50%       14.454200
# 75%       31.000000
# max      512.329200
# Name: 票价, dtype: float64.
text['父母子女个数'].describe()
# count    891.000000  也不存在缺失值
# mean       0.381594
# std        0.806057
# min        0.000000
# 25%        0.000000
# 50%        0.000000
# 75%        0.000000   大部分的父母子女都是接近0的
# max        6.000000  父母子女比较多的还是极少数
# Name: 父母子女个数, dtype: float64
s=text.sort_values(by='父母子女个数',ascending=False)['父母子女个数'].reset_index(drop=True)
s[s==0].index[0]  #=213 也即是0-212的213个sample的父母子女数目大于0，而891-213=678个sample的父母子女数目都是0
