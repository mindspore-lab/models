import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

# 读取问题五数据
mountain_data = pd.read_excel('问题五导入数据.xlsx')
city_coords = pd.read_csv('问题五经纬度.csv')

# 定义初始参数
total_time_limit = ms.Tensor(144, ms.float32)  # 144小时
current_time = ms.Tensor(0, ms.float32)
current_cost = ms.Tensor(0, ms.float32)
current_city = ''
visited_cities = []
visited_spots = []
total_spots = ms.Tensor(0, ms.int32)

# 高铁速度和费用
high_speed = ms.Tensor(300, ms.float32)  # km/h
cost_per_km = ms.Tensor(0.5, ms.float32)  # 元/km

# Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = ms.Tensor(6371, ms.float32)  # 地球半径（公里）
    lat1, lon1, lat2, lon2 = map(lambda x: ops.deg2rad(ms.Tensor(x, ms.float32)), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = ops.sin(dlat/2)**2 + ops.cos(lat1) * ops.cos(lat2) * ops.sin(dlon/2)**2
    c = 2 * ops.asin(ops.sqrt(a))
    return R * c

# 找到评分最高的山的每个城市
unique_cities = mountain_data['来源城市'].unique()
best_mountains = pd.DataFrame()

for city in unique_cities:
    city_mountains = mountain_data[mountain_data['来源城市'] == city]
    best_mountain = city_mountains.loc[city_mountains['评分'].idxmax()]
    best_mountains = best_mountains.append(best_mountain, ignore_index=True)

# 初始化贪心算法变量
max_score = ms.Tensor(0, ms.float32)
best_route = []
best_cost = ms.Tensor(float('inf'), ms.float32)
best_time = ms.Tensor(float('inf'), ms.float32)

# 寻找最佳入境城市（选择评分最高的山所在的城市作为初始城市）
initial_city = best_mountains.loc[best_mountains['评分'].idxmax(), '来源城市']
current_city = initial_city
visited_cities.append(current_city)

# 调试信息
print(f'开始旅程，从{initial_city}出发，时间限制为{total_time_limit}小时')

while current_time < total_time_limit:
    best_city = ''
    best_experience = ms.Tensor(0, ms.float32)
    best_city_time = ms.Tensor(0, ms.float32)
    best_city_cost = ms.Tensor(0, ms.float32)
    best_spot_name = ''
    
    for _, city_row in city_coords.iterrows():
        city = city_row['城市']
        if city not in visited_cities:
            # 计算当前城市到目标城市的距离
            current_city_coords = city_coords[city_coords['城市'] == current_city].iloc[0]
            city_lat, city_lon = current_city_coords['纬度'], current_city_coords['经度']
            
            target_lat, target_lon = city_row['纬度'], city_row['经度']
            
            # 计算距离，转换为公里
            dist = haversine_distance(city_lat, city_lon, target_lat, target_lon)
            travel_time = dist / high_speed  # 转换为小时
            travel_cost = dist * cost_per_km  # 元

            spot_data = best_mountains[best_mountains['来源城市'] == city]
            if spot_data.empty:
                continue
            best_spot = spot_data.iloc[0]
            visit_time = ms.Tensor(best_spot['建议游玩时间'], ms.float32)  # 使用小时
            total_city_time = travel_time + ms.Tensor(0.5, ms.float32) + visit_time  # 加入城市内赶路时间0.5小时
            
            # 加入休息时间的计算
            total_time_with_rest = current_time + total_city_time + ((current_time + total_city_time) // 24) * 8
            
            # 调试信息
            print(f'评估城市: {city}，旅行时间: {travel_time.asnumpy():.2f}小时，游玩时间: {visit_time.asnumpy():.2f}小时，总时间: {total_city_time.asnumpy():.2f}小时，包含休息时间: {total_time_with_rest.asnumpy():.2f}小时')
            
            if total_time_with_rest <= total_time_limit:
                experience = ms.Tensor(best_spot['评分'], ms.float32)
                if experience > best_experience:
                    best_experience = experience
                    best_city = city
                    best_city_time = total_city_time
                    best_city_cost = travel_cost + ms.Tensor(best_spot['门票'], ms.float32)
                    best_spot_name = best_spot['名字']
    
    if not best_city:
        print('未找到适合的下一站，旅程结束')
        break
    
    visited_cities.append(best_city)
    visited_spots.append(best_spot_name)
    current_time = current_time + best_city_time + ((current_time + best_city_time) // 24) * 8  # 更新当前时间，包含休息时间
    current_cost += best_city_cost
    total_spots += ms.Tensor(1, ms.int32)
    current_city = best_city
    
    # 调试信息
    print(f'访问城市: {best_city}（{best_spot_name}），当前总时间: {current_time.asnumpy():.2f}小时，当前总费用: {current_cost.asnumpy():.2f}元')

# 输出结果
print(f'Travel Route: {" -> ".join(visited_cities)}')
print(f'Total Travel Time (hours): {current_time.asnumpy():.2f}')
print(f'Total Cost: {current_cost.asnumpy():.2f}')
print(f'Total Scenic Spots: {total_spots.asnumpy()}')

# 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(city_coords['经度'], city_coords['纬度'], c='blue', marker='.')

# 绘制访问过的城市和路径
for i, city in enumerate(visited_cities):
    city_data = city_coords[city_coords['城市'] == city].iloc[0]
    plt.scatter(city_data['经度'], city_data['纬度'], c='red', marker='o')
    plt.text(city_data['经度'], city_data['纬度'], str(i+1), verticalalignment='bottom', horizontalalignment='right')
    if i > 0:
        prev_city_data = city_coords[city_coords['城市'] == visited_cities[i-1]].iloc[0]
        plt.plot([prev_city_data['经度'], city_data['经度']], 
                 [prev_city_data['纬度'], city_data['纬度']], 'k-')

plt.title('旅行路线')
plt.grid(True)
plt.show()

# 打印总信息
print(f'总旅行时间: {current_time.asnumpy():.2f}小时')
print(f'总费用: {current_cost.asnumpy():.2f}元')
print(f'总景点数量: {total_spots.asnumpy()}')
