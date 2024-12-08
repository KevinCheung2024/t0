# t0
ML-Based Intraday High-Frequency Trading Strategy

文件说明：
·data.parquet是一只股票（603000）半年的高频数据（3s频率）；该数据缺失，需自行导入
·trading_dates.csv记录交易日信息
·trading_dates.py记录日期处理的各类函数
·data_process.py包含数据预处理的各类函数
·split_data_3s.py可将原始数据data.parquet按代码-日期进行拆分
·文件夹raw_data记录拆分后的原始数据
·gen_factor.py可根据拆分后原始数据生成因子值（因子版本为FV001001）
·文件夹factor记录生成的因子值
·model_train.py可根据生成的因子值，训练模型（模型名称为CatBoost）
·文件夹model_tuning记录训练完成的模型（.cb文件）和最优因子组合（.csv文件），一般是经过调参的模型
·model_pred.py根据训练完成的模型进行样本外预测
·文件夹pred_tuning保存预测值，一般是经过调参的模型
·model_crit.py生成模型预测性能
·文件夹figure保存各类图片，包括模型预测性能
·run_strategy.py生成策略交易单（策略版本ST001001）
·文件夹trade_logs记录策略交易单
·strategy_crit.py进行策略评估

运行说明：
·在运行之前，首先将某只股票（603000）一段时间的3s盘口快照数据导入，记为data.parquet,并将所有交易日信息记录在trading_dates.csv中
·在运行之前，上述文件夹需自行创建
·首先运行gen_factor.py,生成因子
·其次运行model_train.py,训练模型
·然后运行model_pred.py,生成预测值
·然后运行model_crit.py,模型性能评估
·接着运行run_strategy.py,生成策略交易单
·最后运行strategy_crit.py,策略评估
