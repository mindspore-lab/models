import pandas as pd
import re
import mindspore as ms
import mindspore.numpy as mnp

# 读取数据，并保留原始列标题
data = pd.read_excel('问题三初始数据.xlsx')

# 提取游玩时间列
time_strings = data['建议游玩时间']

# 初始化游玩时间列表
play_times = []

# 定义处理时间字符串的函数
def process_time_string(time_str):
    # 匹配时间区间
    tokens = re.findall(r'(\d+\.?\d*)小时?\s*-\s*(\d+\.?\d*)小时?', time_str)
    if tokens:
        time_values = ms.Tensor([float(t) for t in tokens[0]], dtype=ms.float32)
        return mnp.mean(time_values)
    
    # 匹配单个时间
    tokens = re.findall(r'(\d+\.?\d*)小时?', time_str)
    if tokens:
        return ms.Tensor(float(tokens[0]), dtype=ms.float32)
    
    # 处理"天"为单位的情况
    tokens = re.findall(r'(\d+\.?\d*)天?\s*-\s*(\d+\.?\d*)天?', time_str)
    if tokens:
        time_values = ms.Tensor([float(t) * 24 for t in tokens[0]], dtype=ms.float32)  # 转换为小时
        return mnp.mean(time_values)
    
    tokens = re.findall(r'(\d+\.?\d*)天', time_str)
    if tokens:
        return ms.Tensor(float(tokens[0]) * 24, dtype=ms.float32)  # 转换为小时
    
    raise ValueError(f'无法解析游玩时间字符串: {time_str}')

# 遍历所有游玩时间字符串并进行处理
for time_str in time_strings:
    try:
        avg_time = process_time_string(time_str)
        play_times.append(avg_time.asnumpy().item())
    except ValueError as e:
        print(e)
        play_times.append(mnp.nan)  # 对于无法解析的时间，添加NaN值

# 将转换后的游玩时间添加到表格中
data['建议游玩时间'] = play_times

# 保存处理后的数据
data.to_csv('游玩时间处理后的数据.csv', index=False)

# 显示前几行以验证结果
print(data.head(10))
