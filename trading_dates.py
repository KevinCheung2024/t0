import pandas as pd
import numpy as np
import os
import datetime
current_path = os.path.dirname(os.path.abspath(__file__))

#导入交易日信息
csv_path = os.path.join(current_path,'trading_dates.csv')
trading_dates_list = pd.read_csv(csv_path,header=None)[0].tolist()

#辅助函数
def get_natual_days(start_date, end_date):
    """
    获取两个日期之间的全部自然日期（包括周末）。
    参数：
    - start_date: 起始日期，格式为 'YYYY-MM-DD'
    - end_date: 结束日期，格式为 'YYYY-MM-DD'
    返回：
    - 一个日期列表，包含所有的自然日期（包括周末）
    """
    # 将字符串转为日期格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # 使用pd.date_range生成所有日期
    all_days = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()
    return all_days


def get_trading_days_in_range(start_date, end_date,dates_list = trading_dates_list):
    """
    获取给定日期区间内的所有交易日
    :param dates_list: 日期列表，格式为 ['YYYY-MM-DD']
    :param start_date: 起始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :return: 给定日期区间内的所有交易日列表
    """
    # 将列表转换为 datetime 格式
    dates = pd.to_datetime(dates_list)

    # 将起始日期和结束日期转换为 datetime 格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date is None:
        start_date = dates.min()
    if end_date is None:
        end_date = dates.max()

    # 筛选区间内的交易日
    trading_days_in_range = dates[(dates >= start_date) & (dates <= end_date)]

    # 返回结果列表
    return trading_days_in_range.strftime('%Y-%m-%d').tolist()

def get_next_trading_day(given_date, dates_list = trading_dates_list):
    """
    获取给定日期的下一个交易日
    :param given_date: 给定日期，格式为 'YYYY-MM-DD'
    :param dates_list: 日期列表，格式为 ['YYYY-MM-DD']
    :return: 给定日期之后的下一个交易日
    """
    # 将列表转换为 datetime 格式
    dates = pd.to_datetime(dates_list)

    # 将给定日期转换为 datetime 格式
    given_date = pd.to_datetime(given_date)

    # 找到所有大于给定日期的交易日
    future_trading_days = dates[dates > given_date]

    # 获取第一个大于给定日期的交易日
    next_trading_day = future_trading_days.min()

    # 返回下一个交易日
    return next_trading_day.strftime('%Y-%m-%d')

def is_sunday(date_str):
    """
    判断输入的日期是否为周日。
    参数：
    - date_str: 日期字符串，格式为 'YYYY-MM-DD'
    返回：
    - 如果是周日，返回 True；否则返回 False
    """
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return date.weekday() == 6  # .weekday() 返回 6 对应周日

import datetime

# def get_previous_trading_days(date_str, n,trading_days_list=trading_dates_list):
#     """
#     获取给定日期之前的 n 个交易日。
#     参数：
#     - date_str: 给定日期字符串，格式为 'YYYY-MM-DD'
#     - trading_days_list: 包含一段时期内所有交易日的日期列表，格式为 ['YYYY-MM-DD', 'YYYY-MM-DD', ...]
#     - n: 想要获取的前 n 个交易日
#     返回：
#     - 一个包含前 n 个交易日的日期列表
#     """
#     # 将输入的日期字符串转为 datetime 对象
#     date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
#     # 将交易日字符串列表转换为 datetime 对象列表
#     trading_days = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in trading_days_list]
#     # 获取目标日期在交易日列表中的索引位置
#     index = trading_days.index(date)
#     # 获取该日期之前的 n 个交易日
#     previous_trading_days = trading_days[max(0, index - n):index]
#     # 返回日期列表，以字符串形式返回
#     return [day.strftime('%Y-%m-%d') for day in previous_trading_days]

# def get_previous_trading_days(date_str, n, trading_days_list=trading_dates_list):
#     """
#     获取给定日期之前的 n 个交易日。
#     参数：
#     - date_str: 给定日期字符串，格式为 'YYYY-MM-DD'
#     - trading_days_list: 包含一段时期内所有交易日的日期列表，格式为 ['YYYY-MM-DD', 'YYYY-MM-DD', ...]
#     - n: 想要获取的前 n 个交易日
#     返回：
#     - 一个包含前 n 个交易日的日期列表
#     """
#     # 将输入的日期字符串转为 datetime 对象
#     date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
#     # 将交易日字符串列表转换为 datetime 对象列表
#     trading_days = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in trading_days_list]
#     # 获取目标日期在交易日列表中的索引位置
#     index = trading_days.index(date)
#     # 获取目标日期之前的 n 个交易日，确保返回所有的交易日（如果不足 n 个）
#     previous_trading_days = trading_days[max(0, index - n):index]
#     # 返回日期列表，以字符串形式返回
#     return [day.strftime('%Y-%m-%d') for day in previous_trading_days]

def get_previous_trading_days(date_str, n, trading_days_list=trading_dates_list):
    """
    获取给定日期之前的 n 个交易日。
    参数：
    - date_str: 给定日期字符串，格式为 'YYYY-MM-DD'
    - trading_days_list: 包含一段时期内所有交易日的日期列表，格式为 ['YYYY-MM-DD', 'YYYY-MM-DD', ...]
    - n: 想要获取的前 n 个交易日
    返回：
    - 一个包含前 n 个交易日的日期列表
    """
    # 将输入的日期字符串转为 datetime 对象
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    # 将交易日字符串列表转换为 datetime 对象列表
    trading_days = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in trading_days_list]
    # 获取目标日期在交易日列表中的索引位置
    # 如果日期不在交易日列表中，例如周日，找到最近的前一个交易日
    index = None
    for i, day in enumerate(trading_days):
        if day >= date:
            index = i
            break
    # 如果日期不在交易日列表中，返回该日期前的n个交易日
    if index is None:
        index = len(trading_days)
    # 获取目标日期之前的 n 个交易日，确保返回所有的交易日（如果不足 n 个）
    previous_trading_days = trading_days[max(0, index - n):index]
    # 返回日期列表，以字符串形式返回
    return [day.strftime('%Y-%m-%d') for day in previous_trading_days]

def get_previous_sunday(date_str):
    '''
    找到给定日期前的最近一个星期日
    '''
    # 将输入的日期字符串转换为日期对象
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    # 计算当前日期是星期几，星期天是 6
    days_to_sunday = (date.weekday() + 1) % 7  # 计算距离上一个星期天的天数
    previous_sunday = date - datetime.timedelta(days=days_to_sunday)
    # 返回结果，格式为 YYYY-MM-DD
    return previous_sunday.strftime('%Y-%m-%d')
