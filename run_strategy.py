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


class ST001001:

    '''
    基于2014-06-18广发证券研报《深度学习之股指期货日内交易策略——大数据深度学习系列之一》构建日内t0交易策略；
    关于交易成本：卖出收取印花税率0.05%，买卖均收取佣金费率0.01%；
    根据开仓信号对应开多头或者空头；在多头时遇到开空信号则反向建仓，遇到开多信号则保持仓位、更新止损成本价；设置时间止损（收盘前10min）和价格止损；
    暂不考虑策略容量；

    根据预测值，生成交易单
    '''

    def __init__(self,date,pred_path,model_name,version,code,logpath,alpha1=0.04,alpha2 = 0.04,r=-0.05,buy_cost=0.0001,sell_cost=0.0006):
        self.alpha1 = alpha1 #多头开仓信号阈值
        self.alpha2 = alpha2 #空头开仓信号阈值
        self.r = r #止损线
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost

        self.date = date #每次仅处理单日交易数据
        self.pred_path = pred_path #预测值所在路径
        self.model_name = model_name #模型名称
        self.version = version #因子版本号
        self.code = code #对单只股票操作
        self.logpath = logpath #交易单保存路径

        #导入数据
        self.backtestlist = get_previous_trading_days(self.date,10) 
        if len(self.backtestlist) <10:
            return
        
        self.df = self.load_pred_data(start_date=self.date,end_date=self.date) #当日数据
        self.df.index = pd.to_datetime(self.df.index)
        self.close_time = self.df.index[-1]

        #根据前10个交易日信息计算信号阈值
        # self.backtestlist = get_previous_trading_days(self.date,10) 
        # if len(self.backtestlist) <10:
        #     return
        self.bt0 = self.backtestlist[0]
        self.bt1 = self.backtestlist[-1]
        self.backtestdf = self.load_pred_data(start_date=self.bt0,end_date=self.bt1)
        if len(self.backtestdf) == 0:
            return
        self.longthr = self.backtestdf['rise'].quantile(1-self.alpha1) #多头开仓概率阈值
        self.shortthr = self.backtestdf['fall'].quantile(1-self.alpha2) #空头开仓概率阈值
        self.backtestlist = None
        self.bt0 = None
        self.bt1 = None
        self.backtestdf = None

        self.position = 0 #初始化仓位：0-空仓，1-多头，-1-空头
        self.trade_log = [] #记录每对建仓、平仓交易
        self.entry_price = None #初始化入场价格
        self.entry_time = None #初始化入场时间
        self.entry_type = None #初始化入场方式
        self.stop_loss_price = None #初始化止损成本价
        # self.exit_price = None #初始化出场价格
        # self.exit_time = None #初始化出场时间
        # self.exit_type = None #初始化出场方式

        self.run()
        self.trade_df = pd.DataFrame(self.trade_log)

        self.save_dir = os.path.join(self.logpath,'ST001001',f'{self.model_name}',f'{self.version}',f'{self.date}')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  # 如果目录不存在，创建目录
        self.file_name = os.path.join(self.save_dir, f'{code}.parquet')
        self.trade_df.to_parquet(self.file_name)


    def load_pred_data(self,start_date,end_date):
        '''
        导入指定日期区间的预测值数据
        '''
        df_list = []
        for dates in get_trading_days_in_range(start_date,end_date):
            pred_dir_path = os.path.join(self.pred_path,f'{self.model_name}',f'{self.version}',f'{dates}',f'{self.code}.parquet')
            if not os.path.exists(pred_dir_path):
                print(f"No Pred Data in {dates}")
                continue
            df = pd.read_parquet(pred_dir_path)
            
            df_list.append(df)
        if len(df_list) == 0:
            return pd.DataFrame()
        total_df = pd.concat(df_list)
        return total_df

    def update_position(self,i,row):
        ask_price1 = row['ask_price1'] #委卖1价
        bid_price1 = row['bid_price1'] #委买1价
        prediction = row['Y1_pred'] #走势预测
        rise_prob = row['rise'] #上涨概率
        fall_prob = row['fall'] #下跌概率
        clock = i #所在时刻

        if self.position == 0: #当前tick空仓
            if prediction == 1 and rise_prob >= self.longthr: #多头信号
                self.position = 1 #建多仓
                self.entry_price = ask_price1*(1+self.buy_cost) #更新入场价格
                self.entry_time = clock #更新入场时间
                self.entry_type = '开多' #更新入场类型
                self.stop_loss_price = ask_price1*(1+self.buy_cost)
                self.trade_log.append({
                    'entry_price': self.entry_price,
                    'entry_time': self.entry_time,
                    'entry_type': self.entry_type,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_type': None,
                    'return': None
                })
            elif prediction == -1 and fall_prob >= self.shortthr: #空头信号
                self.position = -1 #建空仓
                self.entry_price = bid_price1*(1-self.sell_cost) #更新入场价格
                self.entry_time = clock #更新入场时间
                self.entry_type = '开空' #更新入场类型
                self.stop_loss_price = bid_price1*(1-self.sell_cost)
                self.trade_log.append({
                    'entry_price': self.entry_price,
                    'entry_time': self.entry_time,
                    'entry_type': self.entry_type,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_type': None,
                    'return': None
                })
        elif self.position == 1: #当前tick多头持仓
            current_price = bid_price1*(1-self.sell_cost) #当前价
            returns = ((current_price - self.entry_price)/(self.entry_price))*self.position #收益
            if returns <= self.r: #首先判断是否需要止损
                self.position = 0 #平仓
                self.trade_log[-1]['exit_price'] = current_price
                self.trade_log[-1]['exit_time'] = clock
                self.trade_log[-1]['exit_type'] = '止损平多'
                self.trade_log[-1]['return'] = returns
            else: #如果不需要止损
                if prediction == 1 and rise_prob >= self.longthr: #继续触发多头信号，不改变仓位
                    self.stop_loss_price = ask_price1*(1+self.buy_cost) #更新止损成本价
                elif prediction == -1 and fall_prob >= self.shortthr: #空头信号,反向建仓
                    self.position = -1 #平多仓，并空头建仓
                    #首先平仓
                    self.trade_log[-1]['exit_price'] = current_price
                    self.trade_log[-1]['exit_time'] = clock
                    self.trade_log[-1]['exit_type'] = '平多'
                    #current_price = bid_price1 #出场价
                    #returns = ((current_price - self.entry_price)/(self.entry_price))*self.position #收益
                    self.trade_log[-1]['return'] = returns
                    #其次建空头
                    self.entry_price = current_price #下一次入场价
                    self.entry_time = clock #下一次入场时间
                    self.entry_type = '开空' #下一次入场类型
                    self.stop_loss_price = current_price #更新止损成本价
                    self.trade_log.append({
                        'entry_price':self.entry_price,
                        'entry_time':self.entry_time,
                        'entry_type':self.entry_type,
                        'exit_price': None,
                        'exit_time': None,
                        'exit_type': None,
                        'return': None
                    })
        elif self.position == -1: #当前tick空头持仓
            current_price = ask_price1*(1+self.buy_cost) #当前价
            returns = ((current_price - self.entry_price)/(self.entry_price))*self.position #收益
            if returns <= self.r: #首先判断是否需要止损
                self.position = 0 #平仓
                self.trade_log[-1]['exit_price'] = current_price
                self.trade_log[-1]['exit_time'] = clock
                self.trade_log[-1]['exit_type'] = '止损平空'
                self.trade_log[-1]['return'] = returns
            else: #如果不需要止损
                if prediction == -1 and fall_prob >= self.shortthr: #继续触发空头信号，不改变仓位
                    self.stop_loss_price = bid_price1*(1-self.sell_cost) #更新止损成本价
                elif prediction == 1 and rise_prob >= self.longthr: #多头信号,反向建仓
                    self.position = 1 #平空仓，并多头建仓
                    #首先平仓
                    self.trade_log[-1]['exit_price'] = current_price
                    self.trade_log[-1]['exit_time'] = clock
                    self.trade_log[-1]['exit_type'] = '平空'
                    #current_price = ask_price1 #出场价
                    #returns = ((current_price - self.entry_price)/(self.entry_price))*self.position #收益
                    self.trade_log[-1]['return'] = returns
                    #其次建多头
                    self.entry_price = current_price #下一次入场价
                    self.entry_time = clock #下一次入场时间
                    self.entry_type = '开多' #下一次入场类型
                    self.stop_loss_price = current_price #更新止损成本价
                    self.trade_log.append({
                        'entry_price':self.entry_price,
                        'entry_time':self.entry_time,
                        'entry_type':self.entry_type,
                        'exit_price': None,
                        'exit_time': None,
                        'exit_type': None,
                        'return': None
                    })    
        return
    
    def run(self):
        for i,row in self.df.iterrows():
            current_time = i
            time_to_close = self.close_time - current_time
            if time_to_close <= timedelta(minutes=10):
                print(f'于 {current_time} 触发时间止损')
                if self.position ==1:
                    self.trade_log[-1]['exit_price'] = self.trade_log[-1]['entry_price']
                    self.trade_log[-1]['exit_time'] = current_time
                    self.trade_log[-1]['exit_type'] = '时间止损'
                    self.trade_log[-1]['return'] = 0
                break
                '''
                如果考虑时间止损带来的收益，可以加上注释掉上面的代码，用下面的代码：
                '''
                # if self.position ==1:
                #     self.trade_log[-1]['exit_price'] = row['bid_price1']
                #     self.trade_log[-1]['exit_time'] = current_time
                #     self.trade_log[-1]['exit_type'] = '时间止损平多仓'
                #     self.trade_log[-1]['return'] = (row['bid_price1']-self.trade_log[-1]['entry_price'])/self.trade_log[-1]['entry_price']
                # elif self.position == -1:
                #     self.trade_log[-1]['exit_price'] = row['ask_price1']
                #     self.trade_log[-1]['exit_time'] = current_time
                #     self.trade_log[-1]['exit_type'] = '时间止损平空仓'
                #     self.trade_log[-1]['return'] = -(row['ask_price1']-self.trade_log[-1]['entry_price'])/self.trade_log[-1]['entry_price']
                # break
            else:
                self.update_position(i,row)


if __name__ == "__main__":
    start = '2023-07-03'
    end   = '2023-12-29'
    pred_path = os.path.join(current_path,'pred_tuning')
    model_name = 'CatBoost'
    version = 'FV001001'
    codelist = ['603000']
    logpath = os.path.join(current_path,'trade_logs')
    for dates in get_trading_days_in_range(start,end):
        for codes in codelist:
            ST001001(dates,pred_path,model_name,version,codes,logpath)