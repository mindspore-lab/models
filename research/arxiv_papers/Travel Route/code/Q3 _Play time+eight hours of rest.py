import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

# 读取城市数据和景点数据
city_data = pd.read_csv('high_speed_rail.csv')
scenic_data = pd.read_csv('问题三导入数据.csv')

# 添加广州的经纬度
guangzhou_lat = 23.1291
guangzhou_lon = 113.2644

# 定义初始参数
initial_city = '广州'
total_time_limit = ms.Tensor(144, ms.float32)  # 144小时
current_time = ms.Tensor(0, ms.float32)
current_cost = ms.Tensor(0, ms.float32)
current_city = initial_city
visited_cities = [current_city]
total_spots = ms.Tensor(0, ms.int32)

# 高铁速度和费用
high_speed = ms.Tensor(300, ms.float32)  # km/h
cost_per_km = ms.Tensor(0.5, ms.float32)  # 元/km

# Haversine formula to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = ms.Tensor(6371, ms.float32)  # 地球半径（公里）
    lat1, lon1, lat2, lon2 = map(lambda x: mnp.radians(ms.Tensor(x, ms.float32)), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = mnp.sin(dlat/2)**2 + mnp.cos(lat1) * mnp.cos(lat2) * mnp.sin(dlon/2)**2
    c = 2 * mnp.arcsin(mnp.sqrt(a))
    return R * c

# 调试信息
print(f'开始旅程，从{initial_city}出发，时间限制为{total_time_limit}小时')

while current_time < total_time_limit:
    best_city = ''
    best_experience = ms.Tensor(0, ms.float32)
    best_city_time = ms.Tensor(0, ms.float32)
    best_city_cost = ms.Tensor(0, ms.float32)
    
    for _, city_row in city_data.iterrows():
        city = city_row['城市']
        if city not in visited_cities:
            # 计算当前城市到目标城市的距离
            if current_city == '广州':
                city_lat = guangzhou_lat
                city_lon = guangzhou_lon
            else:
                current_city_data = city_data[city_data['城市'] == current_city].iloc[0]
                city_lat = current_city_data['纬度']
                city_lon = current_city_data['经度']
            
            target_lat = city_row['纬度']
            target_lon = city_row['经度']
            
            # 计算距离，转换为公里
            dist = haversine_distance(city_lat, city_lon, target_lat, target_lon)
            travel_time = dist / high_speed  # 转换为小时
            travel_cost = dist * cost_per_km  # 元

            spot_data = scenic_data[scenic_data['来源城市'] == city]
            if spot_data.empty:
                continue
            # 选择评分最高的景点
            best_spot = spot_data.loc[spot_data['评分'].idxmax()]
            visit_time = ms.Tensor(best_spot['建议游玩时间'], ms.float32)  # 使用小时
            total_city_time = travel_time + visit_time
            
            # 加入休息时间的计算
            total_time_with_rest = current_time + total_city_time + ((current_time + total_city_time) // 24) * 8
            
            # 调试信息
            print(f'评估城市: {city}，旅行时间: {travel_time:.2f}小时，游玩时间: {visit_time:.2f}小时，总时间: {total_city_time:.2f}小时，包含休息时间: {total_time_with_rest:.2f}小时')
            
            if total_time_with_rest <= total_time_limit:
                experience = ms.Tensor(best_spot['评分'], ms.float32)
                if experience > best_experience:
                    best_experience = experience
                    best_city = city
                    best_city_time = total_city_time
                    best_city_cost = travel_cost + ms.Tensor(best_spot['门票'], ms.float32)

    if not best_city:
        print('未找到适合的下一站，旅程结束')
        break
    
    visited_cities.append(best_city)
    current_time = current_time + best_city_time + ((current_time + best_city_time) // 24) * 8  # 更新当前时间，包含休息时间
    current_cost += best_city_cost
    total_spots += 1
    current_city = best_city
    
    # 调试信息
    print(f'访问城市: {best_city}，当前总时间: {current_time:.2f}小时，当前总费用: {current_cost:.2f}元')

# 输出结果
print(f'Travel Route: {" -> ".join(visited_cities)}')
print(f'Total Travel Time (hours): {current_time:.2f}')
print(f'Total Cost: {current_cost:.2f}')
print(f'Total Scenic Spots: {total_spots}')

# 可视化结果
plt.figure(figsize=(15, 10))
m = Basemap(llcrnrlat=min(city_data['纬度'])-1, urcrnrlat=max(city_data['纬度'])+1,
            llcrnrlon=min(city_data['经度'])-1, urcrnrlon=max(city_data['经度'])+1,
            resolution='i', projection='merc')

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgreen',lake_color='aqua')

# 绘制所有城市
x, y = m(city_data['经度'].values, city_data['纬度'].values)
m.scatter(x, y, marker='o', color='blue', s=50, zorder=5)

# 绘制访问的城市和路径
for i, city in enumerate(visited_cities):
    city_data_row = city_data[city_data['城市'] == city].iloc[0]
    x, y = m(city_data_row['经度'], city_data_row['纬度'])
    m.scatter(x, y, marker='o', color='red', s=100, zorder=10)
    plt.text(x, y, str(i+1), fontsize=12, ha='right', va='bottom')

    if i > 0:
        prev_city = visited_cities[i-1]
        prev_city_data = city_data[city_data['城市'] == prev_city].iloc[0]
        prev_x, prev_y = m(prev_city_data['经度'], prev_city_data['纬度'])
        m.drawgreatcircle(prev_city_data['经度'], prev_city_data['纬度'], 
                          city_data_row['经度'], city_data_row['纬度'], 
                          linewidth=2, color='black')

# 标记广州
x, y = m(guangzhou_lon, guangzhou_lat)
m.scatter(x, y, marker='o', color='green', s=100, zorder=10)
plt.text(x, y, '广州', fontsize=12, ha='right', va='bottom', color='green')

plt.title('旅行路线')
plt.legend(['未访问城市', '访问城市', '旅行路径', '出发地'], loc='best')
plt.show()
