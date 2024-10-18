import mindspore as ms
import mindspore.numpy as mnp
import pandas as pd
import re

# 读取数据
data = pd.read_csv('门票处理后的数据导入.csv', dtype={'开放时间': str})

# 提取开放时间列
open_time_strings = data["开放时间"].tolist()

# 初始化有效性标志列表和标准开放时间列表
is_valid = ms.Tensor([True] * len(data), ms.bool_)
standard_open_times = [''] * len(data)

# 定义处理单个开放时间字符串的函数
def process_open_time(open_time_str):
    if '已关' in open_time_str or '暂停营业' in open_time_str:
        return False, ''
    
    if re.search(r'\d{1,2}/\d{1,2}', open_time_str):
        return False, ''
    
    if any(day in open_time_str for day in ['周一', '周二', '周三', '周四', '周五', '周六', '周日']):
        return False, ''
    
    tokens = re.findall(r'(\d{1,2}[:：]?\d{2})[^\d]*(\d{1,2}[:：]?\d{2})', open_time_str)
    if tokens:
        start_time, end_time = tokens[0]
        start_time = start_time.replace('：', ':')
        end_time = end_time.replace('：', ':')
        return True, f"{start_time}-{end_time}"
    else:
        return False, ''

# 使用MindSpore的map操作处理开放时间
def process_all_open_times(open_time_strings):
    results = list(map(process_open_time, open_time_strings))
    is_valid = ms.Tensor([r[0] for r in results], ms.bool_)
    standard_open_times = [r[1] for r in results]
    return is_valid, standard_open_times

# 执行处理
is_valid, standard_open_times = process_all_open_times(open_time_strings)

# 过滤数据，只保留有效行
filtered_data = data[is_valid.asnumpy()].copy()
filtered_data["开放时间"] = [time for time, valid in zip(standard_open_times, is_valid.asnumpy()) if valid]

# 保存处理后的数据
filtered_data.to_csv('开放时间处理后的数据.csv', index=False)

# 显示前几行以验证结果
print(filtered_data.head(10))
