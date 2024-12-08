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
from datetime import (datetime,timedelta)
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

'''
策略评价函数
'''
def annualized_return(returns):
    returns.index = pd.to_datetime(returns.index)
    years = (returns.index[-1] - returns.index[0]).days/365
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (1/years) - 1
    return annualized_return
# 年化波动率
def annualized_volatility(returns):
    return returns.std() * np.sqrt(365)
# 夏普比率
def sharpe_ratio(returns, risk_free_rate):
    ann_return = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    return (ann_return - risk_free_rate) / ann_vol
# 最大回撤
def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = peak - cumulative_returns
    maxdraw = drawdown.max()
    drawdown_rate = drawdown/peak
    maxdraw_rate = drawdown_rate.max()
    return {"最大回撤":maxdraw, "最大回撤率":maxdraw_rate}
#回测
def backtest(returns,risk_free_rate):
    dic = {"年化收益率":annualized_return(returns),"年化波动率": annualized_volatility(returns),"夏普比率":sharpe_ratio(returns,risk_free_rate), "最大回撤":max_drawdown(returns)["最大回撤"], "最大回撤率":max_drawdown(returns)["最大回撤率"]}
    return dic

if __name__ == "__main__":
    start = '2023-07-03'
    end   = '2023-12-29'
    strategy_name = 'ST001001'
    model_name = 'CatBoost'
    version_name = 'FV001001'
    code = '603000'
    return_list = []
    date_list = []
    for dates in get_trading_days_in_range(start,end):
        file_dir = os.path.join(current_path,'trade_logs',f'{strategy_name}',f'{model_name}',f'{version_name}',f'{dates}',f'{code}.parquet')
        if not os.path.exists(file_dir):
            continue
        df = pd.read_parquet(file_dir)
        if len(df)==0:
            continue
        returns = (df['return']+1).cumprod().iloc[-1]
        return_list.append(returns)
        date_list.append(dates)
    df = pd.DataFrame([date_list,return_list]).T
    df.columns = ['date','return']
    df.set_index('date',inplace=True)
    print(backtest(df['return']-1,0))

    df['pnl'] = df['return'].cumprod()
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['pnl'], label='PNL', color='b', linewidth=2)
    # plt.fill_between(df.index, df['pnl'], color='skyblue', alpha=0.4)
    plt.title('PNL', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()   
    plt.show() 