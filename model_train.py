from data_process import *
from trading_dates import *
import os
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize
from numpy.linalg import pinv
import time
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import (RFE,SelectFromModel,SelectKBest)
from sklearn.model_selection import (GridSearchCV,train_test_split,cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,\
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool, sum_models
import warnings
warnings.filterwarnings('ignore')
import math
import csv

# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))

#辅助函数
def spearman_score(X,y):
    spearman_scores = [spearmanr(X[:, i], y).correlation for i in range(X.shape[1])]
    return np.abs(spearman_scores)
def save_list_to_csv(list_data, file_path):
    """
    将列表数据保存到CSV文件
    :param list_data: 要保存的列表数据（可以是一个一维或二维列表）
    :param file_path: 保存的文件路径
    """
    # 打开文件并创建 CSV 写入器
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list_data)
        # # 如果是二维列表（例如列表中包含多个子列表）
        # if isinstance(list_data[0], list):
        #     writer.writerows(list_data)  # 写入多行
        # else:
        #     writer.writerow(list_data)  # 写入一行
def save_list_to_json(list_data, file_path):
    """
    将列表数据保存到JSON文件
    :param list_data: 要保存的列表数据
    :param file_path: 保存的文件路径
    """
    with open(file_path, 'w') as json_file:
        json.dump(list_data, json_file)

class CatBoostTrainer: 
    '''
    几点说明：
    1，预测目标：价格走势为Y1，未来60s收益率为Y2
    '''
    def __init__(self,version,start_date,end_date,codelist,factor_path,model_path,n_features = 10,train_test_split=0.2,batch_size=2,random_seed=42,tuning_parms=False):
        self.version = version
        self.start_date = start_date
        self.end_date = end_date
        self.codelist = codelist
        self.factor_path = factor_path
        self.model_path = model_path
        self.n_features = n_features
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.data_loader = DataLoader(self.version,self.start_date,self.end_date,self.codelist,self.batch_size,self.factor_path)
        self.datelist = get_trading_days_in_range(start_date,end_date)
        self.combinations = [(date,code) for date in self.datelist for code in self.codelist]
        #print('Start Selecting Features')
        self.FeatureSelectionSKB()
        print('Feature Selection Finished')
        self.best_params=None
        if tuning_parms is True:
            self.GridSearch()
            print('Params Tuning Finished')
        self.train()
        print('Training Data Finished')

    def FeatureSelectionSKB(self):
        totaldf = self.data_loader.data1
        X = totaldf.drop(['Y1','Y2'],axis=1)
        y = totaldf['Y2']
        selector = SelectKBest(score_func=spearman_score,k=self.n_features)
        X_selected = selector.fit_transform(X,y)
        selected_features = X.columns[selector.get_support()]
        self.bestfactorlist = selected_features
        model_dir = os.path.join(self.model_path, 'CatBoost',self.version)
        os.makedirs(model_dir, exist_ok=True)
        bf_save_path = os.path.join(model_dir, f'best_features_{self.start_date}_{self.end_date}.csv')
        print(self.bestfactorlist)
        save_list_to_csv(self.bestfactorlist,bf_save_path)
        return
    def FeatureSelectionRFE(self,model = GradientBoostingClassifier(),cv=10):
        totaldf = self.data_loader.data1
        # totaldf.reset_index(drop=True,inplace=True)
        # totaldf.drop(['date','code'],axis=1,inplace=True)
        X = totaldf.drop(['Y1'],axis=1)
        y = totaldf['Y1']
        if self.n_features is None:
            cv_scores = []
            best_n_features = None
            best_features = None
            for n_features in range(1, X.shape[1] + 1):             # 遍历不同特征数量
                rfe = RFE(estimator=model, n_features_to_select=n_features)                # 创建RFE模型并拟合
                rfe.fit(X, y)
                X_rfe = rfe.transform(X)
                score = cross_val_score(model, X_rfe, y, cv=cv).mean()                # 使用交叉验证评估性能
                cv_scores.append(score)
                if score == max(cv_scores):  # 记录最优特征组合
                    best_n_features = n_features
                    best_features = X.columns[rfe.support_]
            self.bestfactorlist = best_features  # 返回最优特征数量和特征组合
        if self.n_features is not None:
            rfe = RFE(estimator=model, n_features_to_select=self.n_features)
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_]            # 获取最优特征组合
            self.bestfactorlist = selected_features  # 返回最优特征数量和特征组合
        model_dir = os.path.join(self.model_path, 'CatBoost',self.version)
        os.makedirs(model_dir, exist_ok=True)
        bf_save_path = os.path.join(model_dir, f'best_features_{self.start_date}_{self.end_date}.json')
        save_list_to_json(self.bestfactorlist,bf_save_path)
        return

    def GridSearch(self):
        """
        通过GridSearch调参，自动调节学习率、树深度和正则化系数三个超参数，
        调优后将最佳参数存入self.best_params中。
        """
        totaldf = self.data_loader.data1
        # totaldf.reset_index(drop=True,inplace=True)
        # totaldf.drop(['date','code'],axis=1,inplace=True)
        #X = totaldf.drop(['Y1','Y2'],axis=1)
        feat_cols = self.bestfactorlist
        X = totaldf[feat_cols]
        y = totaldf['Y1']
        # 定义需要调节的超参数空间
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
            'depth': [4, 5, 6, 7],  # 树深度
            #'l2_leaf_reg': [1, 3, 5, 10]  # 正则化系数
        }
        # 创建一个基础的 CatBoost 模型
        model = CatBoostClassifier(iterations=500, random_seed=self.random_seed)
        # 创建 GridSearchCV 对象，进行超参数搜索
        grid_search = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                cv=5,  # 交叉验证的折数
                                scoring='accuracy', 
                                verbose=1)  # 输出进度信息
        # 执行 GridSearchCV，输入训练数据 X 和 y
        # 假设self.X和self.y是你预处理好的训练数据和标签
        grid_search.fit(X, y)
        # 获取调优后的最佳参数
        best_params = grid_search.best_params_
        # 将最佳参数存入 self.best_params
        self.best_params = best_params
        return 

    def train(self):
        combinations = self.combinations
        np.random.shuffle(combinations)
        n_batches = len(combinations) // self.batch_size + 1*(len(combinations) % self.batch_size != 0)
        batches = self.data_loader.gen_dataset()
        if self.best_params is None:
            params = {
                'learning_rate': 0.02,
                'depth': 5,
                'iterations': 500,  # 默认的迭代次数
                'random_seed': self.random_seed
            }
        else:
            params = {**self.best_params, 
                    'iterations': 500, 
                    'random_seed': self.random_seed}
        print(f'{datetime.now()}, total {n_batches} batches...')        
        models_3s = []
        for (i,data) in enumerate(batches):
            if len(data)==0:
                print('No data')
                continue
            feat_cols = self.bestfactorlist
            train_mask = (np.random.rand(len(data)) > self.train_test_split)
            val_mask = np.logical_not(train_mask)            
            train_X = data[feat_cols].iloc[train_mask]
            val_X = data[feat_cols].iloc[val_mask]
            train_y = data['Y1'].iloc[train_mask]
            val_y = data['Y1'].iloc[val_mask]            
            train_data = Pool(data=train_X, label=train_y)
            val_data = Pool(data=val_X, label=val_y)
            if len(models_3s) > 0:
                train_data.set_baseline(models_3s[-1].predict_proba(train_data))
                val_data.set_baseline(models_3s[-1].predict_proba(val_data))
            model = CatBoostClassifier(**params)
            model.fit(train_data, eval_set=val_data, early_stopping_rounds=10)
            models_3s.append(model)
        model = sum_models(models_3s)
        model_dir = os.path.join(self.model_path, 'CatBoost',self.version)
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f'model_{self.start_date}_{self.end_date}.cb')
        model.save_model(model_save_path)
        return

        

class DataLoader:
    def __init__(self, version,start_date, end_date, codelist, batch_size,factor_path: str) -> None:
        """        
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param codelist: 股票列表
        :param factor_path: 因子数据存储的根目录路径
        """
        self.version = version
        self.start_date = start_date
        self.end_date = end_date
        self.codelist = codelist
        self.factor_path = factor_path
        self.datelist = get_trading_days_in_range(start_date,end_date)
        self.combinations = [(date,code) for date in self.datelist for code in self.codelist]
        self.batchsize = batch_size
        self.data = None
        self.load_data()
        print('Data Loaded')
        self.data1 = self.data.reset_index(drop=True)
        self.data1 = self.data1.drop(['date','code'],axis=1)

    def load_data(self) -> None:
        """
        根据日期和股票代码加载因子数据
        """
        all_data = []
        for code in self.codelist:  # 遍历每只股票
            for date in self.datelist:  # 遍历每个日期
                # 根据日期和股票代码生成文件路径
                file_path = os.path.join(self.factor_path,self.version,date,f'{code}.parquet')
                if os.path.exists(file_path):
                    df_data = pd.read_parquet(file_path)
                    df_data['date'] = date
                    df_data['code'] = code
                    all_data.append(df_data)    
        if all_data:
            self.data = pd.concat(all_data, axis=0) #更新数据
            self.data = self.data.dropna()  # 去除缺失值
            # self.data.reset_index(drop=True,inplace=True)
            # self.data.drop(['date','code'],axis=1,inplace=True)
        else:
            self.data = pd.DataFrame()
        #print(self.data)

    def read_daily_data(self, date: str, code: str):
        """
        根据日期和股票代码读取对应的因子数据
        :param date: 日期
        :param code: 股票代码
        :return: 日期和股票代码对应的因子数据（DataFrame）
        """
        df_data = self.data[(self.data['date'] == date) & (self.data['code'] == code)]
        return df_data

    def read_combination(self, combinations: list):
        """
        给定股票-日期组合，读取所有对应的数据
        :param combinations: 股票-日期组合的列表 [(date1, code1), (date2, code2), ...]
        :return: 合并后的DataFrame
        """
        data = []
        for date, code in combinations:
            data.append(self.read_daily_data(date, code))
        if len(data) == 0:
            return pd.DataFrame()
        data = pd.concat(data)
        data = data.dropna()  # 去除缺失值
        return data

    # def total_data(self):
    #     if self.data is None:
    #         self.load_data()
    #     return self.data

    def gen_dataset(self):
        """
        按批次生成数据
        
        :param combinations: 股票-日期组合的列表
        :param batch_size: 批次大小
        :yield: 批次数据
        """
        n_batches = len(self.combinations) // self.batchsize + (len(self.combinations) % self.batchsize != 0)
        for i in range(0, len(self.combinations), self.batchsize):
            yield self.read_combination(self.combinations[i:i + self.batchsize])

def day_train(day,version,n_features,codelist,factor_path,model_path,tuning_params):
    '''
    如果是周日，则训练2周的模型
    '''
    if is_sunday(day) is True:
        print(f'start training model of {day}')
        trading_dates_list = get_previous_trading_days(day,10)
        if len(trading_dates_list)<10:
            print('lacking data')
        else:
            start = trading_dates_list[0]
            end = trading_dates_list[-1]
            CatBoostTrainer(version,start,end,codelist,factor_path,model_path,n_features=n_features,tuning_parms=tuning_params)
    return
def train(start, end, version, n_features,codelist, factor_path, model_path, tuning_params):
    natural_dates = get_natual_days(start, end)
    for date in tqdm(natural_dates, desc="Training Progress", unit="day"):
        #print(date)
        #print(is_sunday(date))
        day_train(date, version, n_features,codelist, factor_path, model_path, tuning_params)
        print(f'{date} training complete')
    print("Training complete!")
    return


if __name__ == "__main__":
    # version = 'FV001001'
    # codelist = ['603000']
    # factor_path = os.path.join(current_path,'factor')
    # #model_path  = os.path.join(current_path,'model')
    # model_path  = os.path.join(current_path,'model')
    # n_features = 20
    # tuning_params = False #如果需要调参，可以选择True
    # start = '2023-07-03'
    # end = '2023-12-29'
    # train(start,end,version,n_features,codelist,factor_path,model_path,tuning_params)

    version = 'FV001001'
    codelist = ['603000']
    factor_path = os.path.join(current_path,'factor')
    #model_path  = os.path.join(current_path,'model')
    model_path  = os.path.join(current_path,'model_tuning')
    n_features = 20
    tuning_params = True #如果需要调参，可以选择True
    start = '2023-07-03'
    end = '2023-12-29'
    train(start,end,version,n_features,codelist,factor_path,model_path,tuning_params)

