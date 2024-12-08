from data_process import *
from trading_dates import *
from model_train import DataLoader
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

def gen_pred(model_name,version,start,end,codelist,model_path,data_path,factor_path,pred_path,model=CatBoostClassifier()):
    for dates in get_trading_days_in_range(start,end):
        previous_training_date = get_previous_sunday(dates)
        trading_dates_list = get_previous_trading_days(previous_training_date,10)
        if len(trading_dates_list) <10:
            continue
        start_date = trading_dates_list[0]
        end_date = trading_dates_list[-1]
        model_file_path = os.path.join(model_path, f'{model_name}',version)
        model_dir = os.path.join(model_file_path,f'model_{start_date}_{end_date}.cb')
        bf_dir = os.path.join(model_file_path,f'best_features_{start_date}_{end_date}.csv')
        print(model_dir)
        print(bf_dir)
        if not os.path.isfile(model_dir):
            print(f'lacking model for date {dates}')
            continue
        bf = pd.read_csv(bf_dir)
        bf = list(bf.columns)
        batch_size=2 #取值不影响结果
        data = DataLoader(version,dates,dates,codelist,batch_size,factor_path).data
        X = data[bf]
        y = data['Y1']
        models = model.load_model(model_dir) 
        y_prob = pd.DataFrame(models.predict_proba(X))
        y_prob.index = data.index
        y_prob.columns = ['fall','constant','rise']
        y_pred = pd.DataFrame(models.predict(X))
        y_pred.index = data.index
        y_pred.columns = ['Y1_pred']
        Ys = pd.concat([y_prob,y_pred,y],axis=1)
        codes = codelist[0] #若仅有一只股票，暂且这样处理
        stock_path = os.path.join(data_path, f'{dates}/{codes}.parquet')
        stock_df = pd.read_parquet(stock_path)
        stock_df.set_index('clock',inplace = True)
        stock_df = pd.concat([stock_df,Ys],axis=1)
        #print(stock_df)
        pred_file_path = os.path.join(pred_path,f'{model_name}',f'{version}',f'{dates}')
        os.makedirs(pred_file_path,exist_ok=True)
        pred_dir_path = os.path.join(pred_file_path,f'{codes}.parquet')
        os.makedirs(pred_file_path, exist_ok=True)
        stock_df.to_parquet(pred_dir_path)
        print(f'{dates} pred saved')
    return

if  __name__ == "__main__":
    # model_name = 'CatBoost'
    # version = 'FV001001'
    # start = '2023-07-03'
    # end = '2023-12-29'
    # codelist = ['603000']
    # model_path  = os.path.join(current_path,'model')
    # pred_path = os.path.join(current_path,'pred')
    # data_path = os.path.join(current_path,'raw_data/3s')
    # factor_path = os.path.join(current_path,'factor')
    # gen_pred(model_name,version,start,end,codelist,model_path,data_path,factor_path,pred_path)

    model_name = 'CatBoost'
    version = 'FV001001'
    start = '2023-07-03'
    end = '2023-12-29'
    codelist = ['603000']
    model_path  = os.path.join(current_path,'model_tuning')
    pred_path = os.path.join(current_path,'pred_tuning')
    data_path = os.path.join(current_path,'raw_data/3s')
    factor_path = os.path.join(current_path,'factor')
    gen_pred(model_name,version,start,end,codelist,model_path,data_path,factor_path,pred_path)
    