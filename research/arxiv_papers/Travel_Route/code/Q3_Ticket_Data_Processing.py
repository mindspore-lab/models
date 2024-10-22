import pandas as pd
import re
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp

# 读取数据
data = pd.read_csv('游玩时间处理后的数据.csv')

# 检查列名
print(data.columns)

# 提取门票列
ticket_strings = data['门票']

# 初始化门票价格列表
ticket_prices = []

# 遍历所有门票字符串并进行处理
for ticket_str in ticket_strings:
    # 检查是否包含价格数字
    has_price = bool(re.search(r'\d+\.?\d*元?', ticket_str))
    
    # 检查是否为免费
    is_free = '免费' in ticket_str
    
    if not has_price and is_free:
        ticket_prices.append(0)
    else:
        # 优先匹配成人票价格
        tokens = re.findall(r'成人票[:：]?\s*¥?(\d+\.?\d*)元?', ticket_str)
        if not tokens:
            # 匹配票价
            tokens = re.findall(r'票价[:：]?\s*¥?(\d+\.?\d*)元?', ticket_str)
        if not tokens:
            # 尝试提取所有价格中的第一个
            tokens = re.findall(r'¥?(\d+\.?\d*)元?', ticket_str)
        
        if tokens:
            ticket_prices.append(float(tokens[0]))
        else:
            ticket_prices.append(np.nan)  # 若未找到价格，则设为NaN

# 将提取出的门票价格转换为MindSpore张量
ms_ticket_prices = ms.Tensor(ticket_prices, dtype=ms.float32)

# 使用MindSpore进行简单的统计计算
valid_prices = mnp.nan_to_num(ms_ticket_prices, 0)
mean_price = mnp.mean(valid_prices)
max_price = mnp.max(valid_prices)
min_price = mnp.min(valid_prices[valid_prices > 0])

print(f"平均票价: {mean_price.asnumpy().item():.2f}")
print(f"最高票价: {max_price.asnumpy().item():.2f}")
print(f"最低票价（不包括免费）: {min_price.asnumpy().item():.2f}")

# 将MindSpore张量转换回Python列表
ticket_prices = ms_ticket_prices.asnumpy().tolist()

# 将提取出的门票价格添加到表格中
data['门票'] = ticket_prices

# 删除包含NaN值的行
data = data.dropna(subset=['门票'])

# 保存处理后的数据
data.to_csv('门票处理后的数据.csv', index=False)

# 显示前几行以验证结果
print(data.head(10))
