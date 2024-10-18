import pandas as pd
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

# 读取问题三数据清洗结果表格
data = pd.read_excel('问题三数据清洗结果.xlsx')

# 定义常见的山景关键词
mountain_keywords = ["山", "峰", "岭", "崖", "峪", "岩", "峦", "寨", "嶂", "谷"]

# 将数据转换为MindSpore张量
names = ms.Tensor(data["名字"].tolist())

# 初始化结果列表
result_indices = []

# 使用MindSpore进行关键词匹配
for keyword in mountain_keywords:
    keyword_tensor = ms.Tensor(keyword)
    contains = ops.strings.string_lower(names).contains(ops.strings.string_lower(keyword_tensor))
    indices = mnp.where(contains)[0]
    result_indices.extend(indices.asnumpy().tolist())

# 移除重复的索引
result_indices = list(set(result_indices))

# 使用索引获取结果数据
mountain_data = data.iloc[result_indices]

# 将结果保存为新的CSV文件
mountain_data.to_csv('问题五初始数据.csv', index=False)

# 显示前几行以验证结果
print(mountain_data.head(10))
