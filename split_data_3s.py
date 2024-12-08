import os
import pandas as pd

# 获取当前脚本的路径
current_path = os.path.dirname(os.path.abspath(__file__))

def split_and_save_parquet(df, date_column, stock_code_column, output_folder):
    """
    将数据按照日期列拆分，并为每个股票代码保存一个 Parquet 文件。
    
    参数:
    df (pd.DataFrame): 需要拆分的原始数据，包含日期列和股票代码列。
    date_column (str): 数据中的日期列名称。
    stock_code_column (str): 数据中的股票代码列名称。
    output_folder (str): 保存 Parquet 文件的文件夹路径。
    
    返回:
    None
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['date'] = df[date_column].dt.date
    
    for date, date_group in df.groupby('date'):
        date_str = date.strftime('%Y-%m-%d')
        date_folder = os.path.join(output_folder, date_str)
        os.makedirs(date_folder, exist_ok=True)

        for stock_code, stock_group in date_group.groupby(stock_code_column):
            stock_code = str(stock_code)
            output_file = os.path.join(date_folder, f'{stock_code}.parquet')
            stock_group.to_parquet(output_file)
            print(f'File saved for {stock_code} on {date_str}: {output_file}')


def main(input_file, date_column, stock_code_column, output_folder):
    """
    主函数，加载数据并按日期和股票代码拆分成多个 Parquet 文件。
    
    参数:
    input_file (str): 输入文件路径
    date_column (str): 日期列名称
    stock_code_column (str): 股票代码列名称
    output_folder (str): 输出文件夹路径
    
    返回:
    None
    """
    df = pd.read_parquet(input_file)  # 读取 parquet 文件
    df[stock_code_column] = df[stock_code_column].apply(lambda x: str(x)[-6:])  # 提取股票代码的后6位
    split_and_save_parquet(df, date_column, stock_code_column, output_folder)


if __name__ == "__main__":
    # 设置输入文件路径，日期列名、股票代码列名和输出文件夹
    input_file = os.path.join(current_path, "data.parquet")  # 输入文件路径
    date_column = 'clock'  # 日期列名
    stock_code_column = 'symbol'  # 股票代码列名
    output_folder = os.path.join(current_path, 'raw_data/3s')  # 输出文件夹路径

    # 调用主函数
    main(input_file, date_column, stock_code_column, output_folder)