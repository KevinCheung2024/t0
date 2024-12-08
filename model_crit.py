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

def model_crit_plot(pred_path,start,end,model_name,version,codelist,figure_path,tuning='Without Tuning'):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    dateslist = get_trading_days_in_range(start,end)
    dateslist_new = []
    for dates in dateslist:
        for codes in codelist: #默认只有一只股票
            pred_file_path = os.path.join(pred_path,f'{model_name}',f'{version}',f'{dates}',f'{codes}.parquet')
            if not os.path.exists(pred_file_path):  # 如果文件不存在，则跳过
                print(f"File not found: {pred_file_path}")
                continue
            pred_df = pd.read_parquet(pred_file_path)
            pred_df.dropna(inplace=True)
            Y = pred_df['Y1']
            Y_pred = pred_df['Y1_pred']
            accr = metrics.accuracy_score(Y,Y_pred)
            precision = metrics.precision_score(Y, Y_pred, average='weighted')
            recall = metrics.recall_score(Y, Y_pred, average='weighted')
            f1 = metrics.f1_score(Y, Y_pred, average='weighted')
            accuracies.append(accr)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            dateslist_new.append(dates)
    plt.figure(figsize=(15, 6))
    plt.plot(dateslist_new, accuracies, label='Accuracy', marker='o')
    plt.plot(dateslist_new, precisions, label='Precision', marker='s')
    plt.plot(dateslist_new, recalls, label='Recall', marker='^')
    plt.plot(dateslist_new, f1s, label='F1 Score', marker='d')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.title(f'Model Evaluation Metrics ({model_name} - {version} -{tuning})')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    dir_path = os.path.join(figure_path, 'Model Evaluation Metrics')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # 如果目录不存在，创建目录
    # 构建文件名并拼接到目录路径
    file_name = f'Model Evaluation Metrics ({model_name} - {version} - {tuning}).png'
    save_path = os.path.join(dir_path, file_name)
    plt.savefig(save_path)
    plt.close()
    return

if __name__ == "__main__":
    # pred_path = os.path.join(current_path,'pred')
    # figure_path = os.path.join(current_path,'figures')
    # start = '2023-07-03'
    # end = '2023-12-29'
    # model_name = 'CatBoost'
    # version = 'FV001001'
    # codelist = ['603000']
    # tuning = 'Without Tuning'
    # model_crit_plot(pred_path,start,end,model_name,version,codelist,figure_path,tuning)

    pred_path = os.path.join(current_path,'pred_tuning')
    figure_path = os.path.join(current_path,'figures')
    start = '2023-07-03'
    end = '2023-12-29'
    model_name = 'CatBoost'
    version = 'FV001001'
    codelist = ['603000']
    tuning = 'Tuning'
    model_crit_plot(pred_path,start,end,model_name,version,codelist,figure_path,tuning)