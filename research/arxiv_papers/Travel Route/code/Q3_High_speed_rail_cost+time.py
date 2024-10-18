import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import great_circle
import mindspore as ms
from mindspore import Tensor, ops

# 定义广州的经纬度
guangzhou_coords = (23.1291, 113.2644)

# 前50个城市及其经纬度信息
cities = [
    ('福州', 26.0745, 119.2965),
    ('汕尾', 22.7877, 115.3751),
    ('安庆', 30.5086, 117.0472),
    # ... 其他城市数据 ...
    ('保亭', 18.6011, 109.6957)
]

# 转换为DataFrame
city_df = pd.DataFrame(cities, columns=['城市', '纬度', '经度'])

# 使用MindSpore计算从广州到其他城市的距离
ms.set_context(mode=ms.PYNATIVE_MODE)

def calculate_distance(lat1, lon1, lat2, lon2):
    return Tensor(great_circle((lat1, lon1), (lat2, lon2)).km)

guangzhou_lat = Tensor(guangzhou_coords[0])
guangzhou_lon = Tensor(guangzhou_coords[1])

lats = Tensor(city_df['纬度'].values)
lons = Tensor(city_df['经度'].values)

distances = ops.map_fn(lambda x: calculate_distance(guangzhou_lat, guangzhou_lon, x[0], x[1]),
                       (lats, lons))

city_df['距离'] = distances.asnumpy()

# 假设高铁速度为300 km/h，高铁费用为0.5元/km
high_speed = Tensor(300.0)  # km/h
cost_per_km = Tensor(0.5)  # 元/km

# 使用MindSpore计算高铁时间和费用
distances_tensor = Tensor(city_df['距离'].values)
city_df['高铁时间'] = (distances_tensor / high_speed).asnumpy()
city_df['高铁费用'] = (distances_tensor * cost_per_km).asnumpy()

# 添加出发城市信息
city_df['出发城市'] = '广州'

# 保存为CSV文件
city_df.to_csv('high_speed_rail.csv', index=False)

# 绘制高铁时间的柱状图
plt.figure(figsize=(12, 6))
plt.bar(city_df['城市'], city_df['高铁时间'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('城市')
plt.ylabel('高铁时间（小时）')
plt.title('广州到各城市的高铁时间')
plt.tight_layout()
plt.grid(True)
plt.show()

# 绘制高铁费用的柱状图
plt.figure(figsize=(12, 6))
plt.bar(city_df['城市'], city_df['高铁费用'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('城市')
plt.ylabel('高铁费用（元）')
plt.title('广州到各城市的高铁费用')
plt.tight_layout()
plt.grid(True)
plt.show()
