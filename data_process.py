import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split
import time
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,\
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import math

# 数据预处理
class DataProcess:
    def __init__(self, df, winsor_level=0.025, tick='clock', columns=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.data = df.copy()
        self.columns = columns if columns is not None else df.columns
        self.tick = tick
        self.winsor_level = winsor_level
    
    def histplot(self):
        # 对某个对象绘制分布图
        for col in self.columns:
            sns.distplot(self.data[col])
            plt.xlabel(f"{col}")
            plt.ylabel("Density")
            plt.show()
        return
    def drop_index(self,threshold=0.01):
        #删除缺失值比例超过1%的指标
        df = self.data.copy()
        nan_ratio = df.isna().mean()
        columns_to_drop = nan_ratio[nan_ratio > threshold].index
        cleaned_df = df.drop(columns=columns_to_drop)
        return cleaned_df
    
    def intertempolate_winsor(self):
        # 将所有 int 类型和 float 类型指标进行缺失值填充和缩尾处理，winsor_level 为缩尾水平
        float_list = self.data.select_dtypes(include=['float', 'int']).columns.tolist()
        if not float_list:
            raise ValueError("No float or int columns found in data")
        data_copy = self.data.copy()
        for col in float_list:
            data_copy[col] = data_copy[col].interpolate(method='linear')  # 线性插值填充
            data_copy[col] = winsorize(data_copy[col], limits=(self.winsor_level, self.winsor_level))
        return data_copy
    
    def to_datetime(self):
        # 将缩尾后数据时间序列转为 datetime 格式
        data_copy = self.intertempolate_winsor().copy()
        data_copy[self.tick] = pd.to_datetime(data_copy[self.tick])
        data_copy.set_index([self.tick], inplace=True)
        return data_copy
    
    def zscore(self):
        #z-标准化
        df = self.data.copy()
        means = df.mean()
        stds = df.std()
        stad_df = (df-means)/stds
        return stad_df
