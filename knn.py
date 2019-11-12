# K近邻算法

# 导入相关库文件
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据集
dataset = pd.read_csv('Social_Network_Ads.csv')
# 获取第3到4列做为（Age、EstimatedSalary）数据作为X
X = dataset.iloc[:, [2, 3]].values
# 获取第5列（Purchased）作为Y
y = dataset.iloc[:, 4].values

# 划分数据为训练集和测试集
from sklearn.model_selection import train_test_split
"""
    train_test_split(train_data,train_target,test_size=0.4, random_state=0,stratify=y_train)
    Parameters：
        train_data：所要划分的样本特征集
        train_target：所要划分的样本结果
        test_size：样本占比，如果是整数的话就是样本的数量
        random_state：是随机数的种子。
        随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
        比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。
        但填0或不填，每次都会不一样。
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 特征缩放
from sklearn.preprocessing import StandardScaler
"""
    StandardScaler(copy=True, with_mean=True, with_std=True):
        Parameters：
            copy ： 默认为true，不修改原来的值，创建副本
            with_mean：  在处理sparse CSR或者 CSC matrices 一定要设置False不然会超内存
            with_std : 布尔值，默认为True，将数据进行缩放
        Attributes：
            scale_： 缩放比例，同时也是标准差
            mean_： 每个特征的平均值
            var_:每个特征的方差
            n_sample_seen_:样本数量，可以通过patial_fit 增加
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test = sc.transform(X_test.astype(np.float))

# 使用训练集训练KNN
from sklearn.neighbors import KNeighborsClassifier
'''
        class KNeighborsClassifier(NeighborsBase, KNeighborsMixin,
                                    SupervisedIntegerMixin, ClassifierMixin):
            Parameters:
                n_neighbors:   默认邻居的数量
                weights：      权重
                    可选参数
                    uniform:    统一的权重. 在每一个邻居区域里的点的权重都是一样的。
                    distance:   权重点等于他们距离的倒数。使用此函数，更近的邻居对于所预测的点的影响更大
                    [callable]: 一个用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组。
                algorithm：    采用的算法
                    可选参数
                     ball_tree: 使用算法 BallTree
                     kd_tree:   使用算法 KDTree
                     brute:     使用暴力搜索
                     auto:      会基于传入fit方法的内容，选择最合适的算法。     
                p:              距离度量的类型
                metric：        树的距离矩阵
                metric_params： 矩阵参数
                n_jobs：        用于搜索邻居，可并行运行的任务数量
'''
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred = classifier.predict(X_test)

# 创建混淆矩阵
from sklearn.metrics import confusion_matrix
"""
    def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
        Parameters：
             y_true:       样本真实分类结果
             y_pred:       样本预测分类结果 
             labels:       给出的类别
             sample_weigh: 样本权重

"""
# 所有正确预测的结果都在对角线上，非对角线上的值为预测错误数量
cm = confusion_matrix(y_test, y_pred)

print(cm)