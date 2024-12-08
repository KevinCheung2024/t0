from data_process import *
from trading_dates import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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

# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))

#因子构建
class FV001001:
    def __init__(self,dataframe):
        self.dataframe = dataframe
    class to_blocks:
        #将ticks数据每10个分一组，构成一个blocks
        def __init__(self,df,block_windows=10):
            self.data = df[["current","volume"]].copy()
            self.block_size = block_windows
        def aggregate_block(self,block):
            open_price = block['current'].iloc[0]
            close_price = block['current'].iloc[-1]
            high_price = block['current'].max()
            low_price = block['current'].min()
            total_volume = block['volume'].sum()
            return pd.Series({'open': open_price,
                              'close': close_price,
                              'high': high_price,
                              'low': low_price,
                              'total_volume': total_volume
                             })
        def aggregate_ticks_with_time(self):
            data = self.data.copy()
#            data.set_index('clock',inplace =True)
#            data.index = pd.to_datetime(data.index)
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                raise ValueError("DataFrame 的索引必须是 datetime 类型")
            data['date'] = data.index.date
            data['block'] = data.groupby('date').cumcount() // self.block_size
            aggregated_data = data.groupby(['date', 'block']).apply(self.aggregate_block)
            aggregated_data['time'] = data.groupby(['date', 'block']).apply(lambda x: x.index[0])
            return aggregated_data.reset_index(drop=True).set_index('time')
    
    class GroupI:
        def __init__(self,df,levels=10):
            self.data=df.copy()
            self.levels = levels
            self.ask_price1 = self.data['ask_price1']
            self.bid_price1 = self.data['bid_price1']
            self.ask_volume1 = self.data['ask_volume1']
            self.bid_volume1 = self.data['bid_volume1']
            self.ask_price = self.data[[f'ask_price{i}' for i in range(1, levels+1)]]
            self.bid_price = self.data[[f'bid_price{i}' for i in range(1, levels+1)]]
            self.ask_volume = self.data[[f'ask_volume{i}' for i in range(1, levels+1)]]
            self.bid_volume = self.data[[f'bid_volume{i}' for i in range(1, levels+1)]]
            self.TimeRange = df.index
#         def basic_set(self):
#             return pd.concat([self.ask_price,self.bid_price,self.ask_volume,self.bid_volume],axis=1)
        def time_insensitive_set(self):
            levels = self.levels
            #bid_ask_spread
            bid_ask_spread = pd.DataFrame(np.array(self.ask_price) - np.array(self.bid_price))
            bid_ask_spread.columns = [f'ask_bid_spread{i}' for i in range(1, levels+1)]
            bid_ask_spread.index = self.TimeRange
            #mid_price
            mid_price = pd.DataFrame((np.array(self.ask_price) + np.array(self.bid_price))/2)
            mid_price.columns = [f'mid_price{i}' for i in range(1, levels+1)]
            mid_price.index = self.TimeRange
            #price_differences
            PD_I = np.array(self.ask_price)[:,-1]- np.array(self.ask_price)[:,0]
            PD_II = np.array(self.bid_price)[:,0]- np.array(self.bid_price)[:,-1]
            PD_III = self.ask_price.diff(axis=1).iloc[:,1:]
            PD_III.columns = [f'ask_price_diff{i}' for i in range(1,levels)]
            PD_IV = self.bid_price.diff(axis=1).iloc[:,1:]
            PD_IV.columns = [f'bid_price_diff{i}' for i in range(1,levels)]
            PD = pd.concat([PD_III,PD_IV],axis=1)
            PD["ask_price_diff_sum"] = PD_I
            PD["bid_price_diff_sum"] = PD_II
            #mean prices and volumes
            MEAN = pd.DataFrame({"ask_price_mean":self.ask_price.mean(axis=1),
                                 "bid_price_mean":self.bid_price.mean(axis=1),
                                 "ask_volume_mean":self.ask_volume.mean(axis=1),
                                 "bid_volume_mean":self.bid_volume.mean(axis=1)})
            #accumulated differences
            bid_ask_volume = pd.DataFrame(np.array(self.ask_volume) - np.array(self.bid_volume))
            bid_ask_volume.index = self.TimeRange
            ACD = pd.DataFrame({"accu_diff_price":bid_ask_spread.sum(axis=1),
                                "accu_diff_volume":bid_ask_volume.sum(axis=1)})
            return pd.concat([bid_ask_spread,mid_price,PD,MEAN,ACD],axis=1)
        def time_sensitive_set(self):
            #price and volume derivatives
            Derivatives = pd.concat([self.ask_price.diff(),self.bid_price.diff(),self.ask_volume.diff(),self.bid_volume.diff()],axis=1)
            Derivatives = Derivatives.rename(columns={col: f'diff_{col}' for col in Derivatives.columns})
            return Derivatives
            #由于3s数据缺乏level 2信息，v7,v8,v9无法计算
        def additional_set(self):
            levels = self.levels
            self.bid_price1 = pd.to_numeric(self.bid_price1, errors='coerce').fillna(0)
            self.ask_price1 = pd.to_numeric(self.ask_price1, errors='coerce').fillna(0)
            self.bid_volume1 = pd.to_numeric(self.bid_volume1, errors='coerce').fillna(0)
            self.ask_volume1 = pd.to_numeric(self.ask_volume1, errors='coerce').fillna(0)
            judgePB1 = (self.bid_price1 >= self.bid_price1.shift(1)).astype(int).fillna(0)
            judgePB2 = (self.bid_price1 == self.bid_price1.shift(1)).astype(int).fillna(0)
            deltaVB = (self.bid_volume1 - self.bid_volume1.shift(1) * judgePB2) * judgePB1
            deltaVB = deltaVB.fillna(0) 
            judgePA1 = (self.ask_price1 <= self.ask_price1.shift(1)).astype(int).fillna(0)
            judgePA2 = (self.ask_price1 == self.ask_price1.shift(1)).astype(int).fillna(0)
            deltaVA = (self.ask_volume1 - self.ask_volume1.shift(1) * judgePA2) * judgePA1
            deltaVA = deltaVA.fillna(0)  # 填充NaN为0
            OI = pd.DataFrame(deltaVA - deltaVB)  # Shen(2015)
            OI.columns = ['OI']
            QR = pd.DataFrame((np.array(self.bid_volume) - np.array(self.ask_volume)) / (np.array(self.bid_volume) + np.array(self.ask_volume)))
            QR = QR.fillna(0)  # 填充NaN为0
            QR.columns = [f'QR{i}' for i in range(1, levels+1)]
            QR.index = self.TimeRange  # Cao & Hansch(2009) 深度不平衡指标
            PDB = -self.bid_price.diff(axis=1).iloc[:, 1:]
            PDA = -self.ask_price.diff(axis=1).iloc[:, 1:]
#             HR = pd.DataFrame((np.array(PDB) - np.array(PDA)) / (np.array(PDB) + np.array(PDA)))
#             HR = HR.fillna(0)  # 填充NaN为0
#             HR.columns = [f'HR{i}' for i in range(2, self.levels+1)]
#             HR.index = self.TimeRange  # Cao & Hansch(2009) 宽度不平衡指标
            Imbalance = pd.concat([OI, QR], axis=1)
            return Imbalance
#         def additional_set(self):
#             levels = self.levels
#             judgePB1 = (self.bid_price1 >= self.bid_price1.shift(1)).astype(int)
#             judgePB2 = (self.bid_price1 == self.bid_price1.shift(1)).astype(int)
#             deltaVB  = (self.bid_volume1 - self.bid_volume1.shift(1)*judgePB2)*judgePB1
#             judgePA1 = (self.ask_price1 <= self.ask_price1.shift(1)).astype(int)
#             judgePA2 = (self.ask_price1 == self.ask_price1.shift(1)).astype(int)
#             deltaVA  = (self.ask_volume1 - self.ask_volume1.shift(1)*judgePA2)*judgePA1
#             OI = pd.DataFrame(deltaVA - deltaVB)  #Shen(2015),交易订单流不平衡指标
#             OI.columns = ['OI']
#             QR = pd.DataFrame((np.array(self.bid_volume) - np.array(self.ask_volume))/(np.array(self.bid_volume)+np.array(self.ask_volume)))
#             QR.columns = [f'QR{i}' for i in range(1, levels+1)]
#             QR.index = self.TimeRange #Cao & Hansch(2009)深度不平衡指标
#             PDB = -self.bid_price.diff(axis=1).iloc[:,1:]
#             PDA = -self.ask_price.diff(axis=1).iloc[:,1:]
#             HR = pd.DataFrame((np.array(PDB) - np.array(PDA))/(np.array(PDB) + np.array(PDA)))
#             HR.columns = [f'HR{i}' for i in range(2,self.levels+1)]
#             HR.index = self.TimeRange #Cao & Hansch(2009)宽度不平衡指标
#             Imbalance = pd.concat([OI,QR,HR])
#             return Imbalance            
        def GroupITotal(self):
#             return pd.concat([self.basic_set(),self.time_insensitive_set(),self.time_sensitive_set(),self.additional_set()],axis=1)
            return pd.concat([self.time_insensitive_set(),self.time_sensitive_set(),self.additional_set()],axis=1)

    class Y:
        def __init__(self,df,buy_cost=0.0001,sell_cost=0.0001):
            self.data = df
            self.bc = buy_cost
            self.sc = sell_cost
#           买卖均收取佣金费率0.01%。
        def middle_price_trend(self):
            data = self.data.copy()
            data["mid_price"] = (data["ask_price1"]+data["bid_price1"])/2
            data["mid_trend"] = data["mid_price"].pct_change().shift(-1)
            df = data[["mid_trend",'mid_price']].dropna()
            df['Y1'] = 0
            df.loc[df['mid_trend'] > self.bc, 'Y1'] = 1
            df.loc[df['mid_trend'] < -self.sc, 'Y1'] = -1
            # df.loc[df['mid_trend'] > 0, 'Y1'] = 1
            # df.loc[df['mid_trend'] < 0, 'Y1'] = -1
            df['Y2'] = (df['mid_price'].shift(-20)-df['mid_price'])/df['mid_price']
            return df[["Y1","Y2"]]
             
    class Technology_analysis:
        def __init__(self,df):
            self.data = df
        def EMA(self,dataframe,span):
            return dataframe.ewm(span = span, adjust=False).mean()
        def MACD(self,short_span=12, long_span=26, signal_span=9):
            df = self.data.copy()
            df["EMA_short"] =self.EMA(df["close"],short_span)
            df["EMA_long"] = self.EMA(df["close"],long_span)
            df["DIF"] = df["EMA_short"]-df["EMA_long"]
            df["DEA"] = self.EMA(df["DIF"],signal_span)
            df["MACD"] = df["DIF"] - df["DEA"]
            return df[["DIF","DEA","MACD"]]
        def KDJ(self, n=9, k_period=3, d_period=3):
            data = self.data.copy()
            data["low_min"] = data['low'].rolling(window=n, min_periods=1).min()
            data["high_max"] = data['high'].rolling(window=n, min_periods=1).max()
            data["rsv"] = (data['close'] - data["low_min"]) / (data["high_max"] - data["low_min"]) * 100
            data["k_values"] = data["rsv"].ewm(span=k_period, adjust=False).mean()
            data["d_values"] = data["k_values"].ewm(span=d_period, adjust=False).mean()
            data["j_values"] = 3 * data["k_values"] - 2 * data["d_values"]
            return data[["k_values","d_values","j_values"]]
        def WilliamsR(self, n=14):
            data = self.data.copy()
            data["high_max"] = data['high'].rolling(window=n, min_periods=1).max()
            data["low_min"] = data['low'].rolling(window=n, min_periods=1).min()
            data["%R"] = (data["high_max"] - data["close"]) / (data["high_max"] - data["low_min"]) * (-100)
            return data["%R"]
        def RSI(self, period=14):
            data = self.data.copy()
            data['change'] = data['close'].diff()
            data['gain'] = data['change'].apply(lambda x: x if x > 0 else 0)
            data['loss'] = data['change'].apply(lambda x: -x if x < 0 else 0)
            data['avg_gain'] = data['gain'].rolling(window=period, min_periods=1).mean()
            data['avg_loss'] = data['loss'].rolling(window=period, min_periods=1).mean()
            data['rs'] = data['avg_gain'] / data['avg_loss']
            data['rsi'] = 100 - (100 / (1 + data['rs']))
            return data[['rsi']]
        def CR(self, period=26):
            data = self.data.copy()
            data['MP'] = (data['high'] + data['low']) / 2
            data['H-MP'] = data['high'] - data['MP']
            data['MP-L'] = data['MP'] - data['low']
            data['sum_H-MP'] = data['H-MP'].rolling(window=period, min_periods=1).sum()
            data['sum_MP-L'] = data['MP-L'].rolling(window=period, min_periods=1).sum()
            data['CR'] = data['sum_H-MP'] / data['sum_MP-L'] * 100
            return data[['CR']]
        def CCI(self, period=20):
            data = self.data.copy()
            data['TP'] = (data['high'] + data['low'] + data['close']) / 3
            data['MA'] = data['TP'].rolling(window=period, min_periods=1).mean()
            data['MD'] = data['TP'].rolling(window=period, min_periods=1).apply(lambda x: (x - x.mean()).abs().mean())
            data['CCI'] = (data['TP'] - data['MA']) / (0.015 * data['MD'])
            return data[['CCI']]
        def TOWER(self):
            data = self.data.copy()
            data['PriceChange'] = data['close'] - data['open']
            data['PriceRange'] = data['high'] - data['low']
            data['TOWER'] = data['PriceChange'] / (data['PriceRange'] * data['total_volume'])
            return data[['TOWER']]
        def MTM(self,period=10):
            data = self.data.copy()
            data['MTM'] = data['close'] - data['close'].shift(period)
            return data[['MTM']]
        def BB(self, window=20, num_std=2):
            df = self.data.copy()
            df['SMA'] = df['close'].rolling(window=window).mean()
            df['STD'] = df['close'].rolling(window=window).std()
            df["Middel Band"] = df['SMA']
            df['Upper Band'] = df['SMA'] + (df['STD'] * num_std)
            df['Lower Band'] = df['SMA'] - (df['STD'] * num_std)
            return df[["Middel Band",'Upper Band','Lower Band']]       
        def TRIX(self, period=15):
            data = self.data.copy()
            data['EMA1'] = data['close'].ewm(span=period, adjust=False).mean()
            data['EMA2'] = data['EMA1'].ewm(span=period, adjust=False).mean()
            data['EMA3'] = data['EMA2'].ewm(span=period, adjust=False).mean()
            data['TRIX'] = (data['EMA3'] - data['EMA3'].shift(1)) / data['EMA3'].shift(1) * 100
            return data[['TRIX']]
        def DMI(self, period=14):
            data = self.data.copy()
            data['H-L'] = data['high'] - data['low']
            data['H-PC'] = abs(data['high'] - data['close'].shift(1))
            data['L-PC'] = abs(data['low'] - data['close'].shift(1))
            data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            data['+DM'] = data['high'].diff()
            data['-DM'] = -data['low'].diff()
            data['+DM'] = data['+DM'].where((data['+DM'] > data['-DM']) & (data['+DM'] > 0), 0.0)
            data['-DM'] = data['-DM'].where((data['-DM'] > data['+DM']) & (data['-DM'] > 0), 0.0)
            data['TR_14'] = data['TR'].rolling(window=period).sum()
            data['+DM_14'] = data['+DM'].rolling(window=period).sum()
            data['-DM_14'] = data['-DM'].rolling(window=period).sum()
            data['+DI'] = 100 * (data['+DM_14'] / data['TR_14'])
            data['-DI'] = 100 * (data['-DM_14'] / data['TR_14'])
            data['DX'] = 100 * abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])
            data['ADX'] = data['DX'].rolling(window=period).mean()
            return data[['+DI', '-DI', 'ADX']]
        def ABV(self):
            data = self.data.copy()
            data['daily_ret'] = data['close'].diff()
            data['direction'] = np.where(data['daily_ret'] > 0, 1, np.where(data['daily_ret'] < 0, -1, 0))
            data['volume_direction'] = data['total_volume'] * data['direction']
            data['OBV'] = 0
            data['OBV'] = data['volume_direction'].cumsum()
            return data[['OBV']]
#         def SAR(self):
#             return
        def MIKE(self,n=10):
            data = self.data.copy()
            data['HighMax'] = data['high'].rolling(window=n).max()
            data['LowMin'] = data['low'].rolling(window=n).min()
            data['Midline'] = (data['HighMax'] + data['LowMin']) / 2
            data['WR'] = data['HighMax']
            data['MR'] = data['Midline'] + (data['HighMax'] - data['Midline'])
            data['SR'] = data['MR'] + (data['HighMax'] - data['LowMin'])
            data['WS'] = data['LowMin']
            data['MS'] = data['Midline'] - (data['Midline'] - data['LowMin'])
            data['SS'] = data['MS'] - (data['HighMax'] - data['LowMin'])
            return data[['Midline', 'WR', 'MR', 'SR', 'WS', 'MS', 'SS']]
        def DMA(self, fast_ma_period=10, slow_ma_period=20):
            data = self.data.copy()
            data['Fast MA'] = data['close'].rolling(window=fast_ma_period).mean()
            data['Slow MA'] = data['close'].rolling(window=slow_ma_period).mean()
            data['DMA'] = data['Fast MA'] - data['Slow MA']
            return data[['DMA']]
        def TAPI(self):
            data = self.data.copy()
            data['VWAP'] = (data['close'] * data['total_volume']).cumsum() / data['total_volume'].cumsum()
            data['Price Change'] = data['close'] - data['VWAP']
            data['TAPI'] = data['Price Change'].cumsum()
            return data[['TAPI']]
        def PSY(self, period=12):
            data = self.data.copy()
            data['Price Change'] = data['close'].diff()
            data['Up Days'] = (data['Price Change'] > 0).rolling(window=period).sum()
            data['PSY'] = (data['Up Days'] / period) * 100
            return data[['PSY']]
        def ARBR(self, n=26):
            data = self.data.copy()
            data['Price Change'] = data['close'].diff()
            data['AR'] = (data['high'] - data['open']) / (data['open'] - data['low']).rolling(window=n).sum() * 100
            data['BR'] = (data['high'] - data['close'].shift(1)) / (data['close'].shift(1) - data['low']).rolling(window=n).sum() * 100
            return data[['AR', 'BR']]
        def VR(self, n=26):
            data = self.data.copy()
            data['Price Change'] = data['close'].diff()
            data['Up Volume'] = data['total_volume'] * (data['Price Change'] > 0)
            data['Down Volume'] = data['total_volume'] * (data['Price Change'] < 0)
            data['VR'] = (data['Up Volume'].rolling(window=n).sum() / data['Down Volume'].rolling(window=n).sum()) * 100
            return data[['VR']]
        def BIAS(self, n=12):
            data = self.data.copy()
            data['MA'] = data['close'].rolling(window=n).mean()
            data['BIAS'] = ((data['close'] - data['MA']) / data['MA']) * 100
            return data[['BIAS']]
#         def OBOS(self, n=14):
#             data = self.data.copy()
#             data['RSI'] = self.RSI(n)['rsi']
# #             data['OBOS'] = data['RSI'].apply(lambda x: 'Overbought' if x > 70 else ('Oversold' if x < 30 else 'Neutral'))
#             data['OBOS'] = data['RSI'].apply(lambda x: 1 if x > 70 else (-1 if x < 30 else 0))
#             return data[['OBOS']]
        def ADR(self, n=20):
            data = self.data.copy()
            data['Daily Range'] = data['high'] - data['low']
            data['ADR'] = data['Daily Range'].rolling(window=n).mean()
            return data[['ADR']]
        def ADL(self):
            data = self.data
            data['CLV'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            data['CLV'].fillna(0, inplace=True)  # 处理分母为 0 的情况
            data['AD'] = data['CLV'] * data['total_volume']
            data['ADL'] = data['AD'].cumsum()
            return data["ADL"]
        def combine_indicators(self):
            df = self.data.copy()
            df_list = []
            macd_df = self.MACD()
            kdj_df = self.KDJ()
            williamsr_df = self.WilliamsR()
            rsi_df = self.RSI()
            cr_df = self.CR()
            cci_df = self.CCI()
            tower_df = self.TOWER()
            mtm_df = self.MTM()
            bb_df = self.BB()
            trix_df = self.TRIX()
            dmi_df = self.DMI()
            abv_df = self.ABV()
            mike_df = self.MIKE()
            dma_df = self.DMA()
            tapi_df = self.TAPI()
            psy_df = self.PSY()
            arbr_df = self.ARBR()
            vr_df = self.VR()
            bias_df = self.BIAS()
            # obos_df = self.OBOS()
            adr_df = self.ADR()
            adl_df = self.ADL()
            df_list.extend([macd_df, kdj_df, mtm_df,
                            bb_df, trix_df, dmi_df, abv_df, mike_df, dma_df, tapi_df, psy_df,
                            arbr_df, vr_df, bias_df, adr_df, adl_df])
            for indicator_df in df_list:
                df = pd.merge(df, indicator_df, left_index=True, right_index=True, how='outer')
            return df
#下面是Ntakaris et al.(2020)中的其他指标：
#         def ADL(self):
#             data = self.data
#             data['CLV'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
#             data['CLV'].fillna(0, inplace=True)  # 处理分母为 0 的情况
#             data['AD'] = data['CLV'] * data['total_volume']
#             data['ADL'] = data['AD'].cumsum()
#             return data["ADL"]
#         def AO(self, fast_period=5, slow_period=34):
#             data = self.data
#             data['Midpoint_Price'] = (data['high'] + data['low']) / 2
#             data['Fast_MA'] = data['Midpoint_Price'].rolling(window=fast_period).mean()
#             data['Slow_MA'] = data['Midpoint_Price'].rolling(window=slow_period).mean()
#             data['AO'] = data['Fast_MA'] - data['Slow_MA']
#             return data['AO']
#         def AC(data, ao_period=5):
#             data = self.data
#             data['AO'] =  self.AO()
#             data['AO_SMA'] = data['AO'].rolling(window=ao_period).mean()
#             data['AC'] = data['AO'] - data['AO_SMA']
#             return data['AC']
#         def ADX(data, period=14):
#             data = self.data
#             data['TR'] = data.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
#             data['+DM'] = (data['high'] - data['high'].shift(1)).where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']), 0)
#             data['-DM'] = (data['low'].shift(1) - data['low']).where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)), 0)
#             data['+TRDI'] = (data['+DM'] / data['TR']) * 100
#             data['-TRDI'] = (data['-DM'] / data['TR']) * 100
#             data['+DI'] = data['+TRDI'].rolling(window=period).mean()
#             data['-DI'] = data['-TRDI'].rolling(window=period).mean()
#             data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100
#             data['ADX'] = data['DX'].rolling(window=period).mean()
#             data['ADXR'] = (data['ADX']+data['ADX'].shift(1))/2
#             return data[['ADX','ADXR']]
#         def SMA(self,window):
#             data = self.data
#             data['Midpoint_Price'] = (data['high'] + data['low']) / 2
#             data[f'SMA{window}'] = data['Midpoint_Price'].rolling(window=window).mean()
#             return data[f'SMA{window}']
#         def DMA(self):
#             data = self.data
#             data["jaw"] = self.SMA(window=13)
#             data["teeth"] = self.SMA(window=8)
#             data["lips"] = self.SMA(window=5)
#             return data[["jaw","teeth","lips"]]
#         def EMA(data, window):
#             data = self.data
#             data['Midpoint_Price'] = (data['high'] + data['low']) / 2
#             data[f"EMA{window}"] = data['Midpoint_Price'].ewm(span=window, min_periods=window, adjust=False).mean()
#             return data[f"EMA{window}"]
#         def APO(self):
#             data = self.data
#             data['APO'] = self.EMA(window=5)-self.EMA(window=13)
#             return data["APO"]
#         def Aroon(data, period=20):
#             data = self.data
#             rolling_high_idx = data['high'].rolling(window=period, min_periods=1).apply(lambda x: period - x.argmax() - 1)
#             rolling_low_idx = data['low'].rolling(window=period, min_periods=1).apply(lambda x: period - x.argmin() - 1)
#             aroon_up = ((period - rolling_high_idx) / period) * 100
#             aroon_down = ((period - rolling_low_idx) / period) * 100
#             data['Aroon_Up'] = aroon_up
#             data['Aroon_Down'] = aroon_down
#             data['Aroon_Oscillator'] = data['Aroon_Up']- data['Aroon_Down']
#             return data[["Aroon_Up","Aroon_Down","Aroon_Oscillator"]]
#         def ATR(self, period=14):
#             data = self.data
#             data['High-Low'] = data['high'] - data['low']
#             data['High-PreviousClose'] = (data['high'] - data['close'].shift()).abs()
#             data['Low-PreviousClose'] = (data['low'] - data['close'].shift()).abs()
#             data['TR'] = data[['High-Low', 'High-PreviousClose', 'Low-PreviousClose']].max(axis=1)
#             data['ATR'] = data['TR'].rolling(window=period, min_periods=1).mean()
#             return data["ATR"]
#         def BB(self, window=20, num_std=2):
#             df = self.data
#             df['SMA'] = df['close'].rolling(window=window).mean()
#             df['STD'] = df['close'].rolling(window=window).std()
#             df["Middel Band"] = df['SMA']
#             df['Upper Band'] = df['SMA'] + (df['STD'] * num_std)
#             df['Lower Band'] = df['SMA'] - (df['STD'] * num_std)
#             return df[["Middel Band",'Upper Band','Lower Band']]
#         def ichimoku_cloud(self):
#             df = self.data
#             high_9 = df['High'].rolling(window=9).max()
#             low_9 = df['Low'].rolling(window=9).min()
#             df['Tenkan-sen'] = (high_9 + low_9) / 2
#             high_26 = df['High'].rolling(window=26).max()
#             low_26 = df['Low'].rolling(window=26).min()
#             df['Kijun-sen'] = (high_26 + low_26) / 2
#             df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)
#             high_52 = df['High'].rolling(window=52).max()
#             low_52 = df['Low'].rolling(window=52).min()
#             df['Senkou Span B'] = ((high_52 + low_52) / 2).shift(26)
#             df['Chikou Span'] = df['Close'].shift(-26)
#             return df[['Tenkan-sen','Kijun-sen','Senkou Span A','Senkou Span B','Chikou Span']]
    
    def output(self):
        # 处理微观结构数据
        micro_structure = self.GroupI(self.dataframe).GroupITotal()
        micro_structure = DataProcess(micro_structure).drop_index()
        micro_structure = DataProcess(micro_structure).intertempolate_winsor()
        micro_structure = DataProcess(micro_structure).zscore()

        # 计算技术指标数据
        data_blocks = self.to_blocks(self.dataframe).aggregate_ticks_with_time()
        technology_index = self.Technology_analysis(data_blocks).combine_indicators()  # 计算技术指标
        technology_index = DataProcess(technology_index).drop_index()
        technology_index = DataProcess(technology_index).intertempolate_winsor()
        technology_index = DataProcess(technology_index).zscore()

        # 计算中位价格走势
        Y1 = self.Y(self.dataframe).middle_price_trend()

        # 合并所有数据
        total_data = pd.concat([micro_structure, technology_index, Y1], axis=1)

        # 填充缺失值并删除最后一行
        total_data = total_data.fillna(method='ffill')
        total_data = total_data.drop(total_data.tail(1).index)

        # 删除不需要的列
        total_data = total_data.drop(["open", "close", "high", "low", "total_volume"], axis=1)

        # 返回最终合并的结果
        return total_data

#计算因子组合

# def calc_Portfolios(version, input_path, output_path, codelist, start_date=None, end_date=None):
#     datelist = get_trading_days_in_range(start_date, end_date)
    
#     for dates in datelist:
#         for codes in codelist:
#             stock_path = os.path.join(input_path, f'{dates}/{codes}.parquet')
            
#             if not os.path.exists(stock_path):
#                 continue  # 跳过不存在的文件
            
#             stock_df = pd.read_parquet(stock_path)
#             stock_df.drop(['clock_int', 'symbol'], axis=1, inplace=True)
            
#             # 动态获取函数并调用
#             if version in globals():
#                 total_factor = globals()[version](stock_df)
#             else:
#                 raise ValueError(f"Function {version} is not defined")
            
#             # 创建输出目录（如果不存在）
#             output_dir = os.path.join(output_path, version, str(dates))
#             os.makedirs(output_dir, exist_ok=True)
            
#             # 保存结果到输出路径
#             factor_path = os.path.join(output_dir, f'{codes}.parquet')
#             total_factor.to_parquet(factor_path)
    
#     return 

def process_stock_data(version, dates, codes, input_path, output_path):
    stock_path = os.path.join(input_path, f'{dates}/{codes}.parquet')
    
    if not os.path.exists(stock_path):
        return None  # 跳过不存在的文件
    
    stock_df = pd.read_parquet(stock_path)
    stock_df.drop(['clock_int', 'symbol'], axis=1, inplace=True)
    stock_df.set_index('clock',inplace=True)
    stock_df.index = pd.to_datetime(stock_df.index)
    
    # 动态获取函数并调用
    if version in globals():
        total_factor = globals()[version](stock_df).output()
    else:
        raise ValueError(f"Function {version} is not defined")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.join(output_path, version, str(dates))
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到输出路径
    factor_path = os.path.join(output_dir, f'{codes}.parquet')
    total_factor.to_parquet(factor_path)
    
    return f"{dates}/{codes} processed"

def calc_Portfolios(version, input_path, output_path, codelist, start_date=None, end_date=None):
    datelist = get_trading_days_in_range(start_date, end_date)
    
    # 创建线程池执行并行任务
    futures = []
    with ThreadPoolExecutor() as executor:
        for dates in datelist:
            for codes in codelist:
                futures.append(
                    executor.submit(process_stock_data, version, dates, codes, input_path, output_path)
                )
        
        # 使用 tqdm 进度条显示任务完成情况
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing stocks"):
            result = future.result()  # 获取执行结果
            if result:
                print(result)  # 打印每个文件处理完成的消息
    
    return "Portfolio calculation completed"


if __name__ == '__main__':
    version = 'FV001001'
    input_path = os.path.join(current_path,'raw_data/3s')
    output_path = os.path.join(current_path,'factor')
    codelist = ['603000']
    start_date = None
    end_date = None
    calc_Portfolios(version, input_path, output_path, codelist, start_date, end_date)