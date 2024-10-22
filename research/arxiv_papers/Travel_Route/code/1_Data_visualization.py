import pandas as pd
import matplotlib.pyplot as plt

# 读取城市经纬度数据
city_coords = pd.read_csv('问题五经纬度.csv')

# 创建地理散点图
plt.figure(figsize=(12, 8))
plt.scatter(city_coords["经度"], city_coords["纬度"], marker='o')
plt.title('问题五城市经纬度分布')
plt.grid(True)

# 添加城市标注
for i, row in city_coords.iterrows():
    plt.annotate(row["城市"], (row["经度"], row["纬度"]), 
                 xytext=(5, 5), textcoords='offset points', 
                 ha='right', va='bottom')

# 设置坐标轴标签
plt.xlabel('经度')
plt.ylabel('纬度')

# 显示图形
plt.show()
