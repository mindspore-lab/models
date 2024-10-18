import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor

# 读取城市数据和景点数据
city_data = pd.read_csv('high_speed_rail.csv')
scenic_data = pd.read_csv('问题三导入数据.csv')

# 添加广州的经纬度
guangzhou_lat = 23.1291
guangzhou_lon = 113.2644

# 定义初始参数
initial_city = '广州'
total_time_limit = 144  # 144小时
current_time = 0
current_cost = 0
current_city = initial_city
visited_cities = [current_city]
visited_spots = []
total_spots = 0

# 高铁速度和费用
high_speed = 300  # km/h
cost_per_km = 0.5  # 元/km

# 使用MindSpore的Tensor和函数重新定义Haversine距离计算
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(lambda x: Tensor(x, ms.float32), [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = mnp.sin(dlat/2)**2 + mnp.cos(lat1) * mnp.cos(lat2) * mnp.sin(dlon/2)**2
    c = 2 * mnp.arctan2(mnp.sqrt(a), mnp.sqrt(1-a))
    return R * c

# 调试信息
print(f'开始旅程，从{initial_city}出发，时间限制为{total_time_limit}小时')

while current_time < total_time_limit:
    best_city = ''
    best_experience = Tensor(0, ms.float32)
    best_city_time = Tensor(0, ms.float32)
    best_city_cost = Tensor(0, ms.float32)
    best_spot_name = ''
    
    for _, city_row in city_data.iterrows():
        city = city_row['城市']
        if city not in visited_cities:
            # 计算当前城市到目标城市的距离
            if current_city == '广州':
                city_lat, city_lon = guangzhou_lat, guangzhou_lon
            else:
                current_city_data = city_data[city_data['城市'] == current_city].iloc[0]
                city_lat, city_lon = current_city_data['纬度'], current_city_data['经度']
            
            target_lat, target_lon = city_row['纬度'], city_row['经度']
            
            # 使用MindSpore的Tensor进行计算
            dist = haversine_distance(city_lat, city_lon, target_lat, target_lon)
            travel_time = dist / Tensor(high_speed, ms.float32)
            travel_cost = dist * Tensor(cost_per_km, ms.float32)

            spot_data = scenic_data[scenic_data['来源城市'] == city]
            if spot_data.empty:
                continue
            # 选择评分最高的景点
            best_spot = spot_data.loc[spot_data['评分'].idxmax()]
            visit_time = Tensor(best_spot['建议游玩时间'], ms.float32)
            local_travel_time = Tensor(0.5, ms.float32)
            total_city_time = travel_time + local_travel_time + visit_time
            
            # 使用MindSpore的算术运算
            total_time_with_rest = current_time + total_city_time + ((current_time + total_city_time) // 24) * 8
            
            # 调试信息
            print(f'评估城市: {city}，旅行时间: {travel_time:.2f}小时，游玩时间: {visit_time:.2f}小时，总时间: {total_city_time:.2f}小时，包含休息时间: {total_time_with_rest:.2f}小时')
            
            if total_time_with_rest <= total_time_limit:
                experience = Tensor(best_spot['评分'], ms.float32)
                if experience > best_experience:
                    best_experience = experience
                    best_city = city
                    best_city_time = total_city_time
                    best_city_cost = travel_cost + Tensor(best_spot['门票'], ms.float32)
                    best_spot_name = best_spot['名字']
    
    if not best_city:
        print('未找到适合的下一站，旅程结束')
        break
    
    visited_cities.append(best_city)
    visited_spots.append(best_spot_name)
    current_time = current_time + best_city_time + ((current_time + best_city_time) // 24) * 8  # 更新当前时间，包含休息时间
    current_cost += best_city_cost
    total_spots += 1
    current_city = best_city
    
    # 调试信息
    print(f'访问城市: {best_city}，当前总时间: {current_time:.2f}小时，当前总费用: {current_cost:.2f}元')

# 输出结果
print(f'Travel Route: {" -> ".join(visited_cities)}')
print(f'Visited Spots: {" -> ".join(visited_spots)}')
print(f'Total Travel Time (hours): {current_time:.2f}')
print(f'Total Cost: {current_cost:.2f}')
print(f'Total Scenic Spots: {total_spots}')

# 可视化结果
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 绘制所有城市
ax.scatter(city_data['经度'], city_data['纬度'], color='blue', s=50, transform=ccrs.PlateCarree(), label='未访问城市')

# 标记访问过的城市
for i, city in enumerate(visited_cities):
    city_data_row = city_data[city_data['城市'] == city].iloc[0]
    ax.scatter(city_data_row['经度'], city_data_row['纬度'], color='red', s=100, transform=ccrs.PlateCarree())
    ax.text(city_data_row['经度'], city_data_row['纬度'], str(i+1), verticalalignment='bottom', horizontalalignment='right', transform=ccrs.PlateCarree())

# 绘制路径
for i in range(len(visited_cities) - 1):
    start_city = visited_cities[i]
    end_city = visited_cities[i+1]
    start_data = city_data[city_data['城市'] == start_city].iloc[0]
    end_data = city_data[city_data['城市'] == end_city].iloc[0]
    
    plt.plot([start_data['经度'], end_data['经度']], [start_data['纬度'], end_data['纬度']], 
             color='black', linewidth=2, transform=ccrs.Geodetic(), label='旅行路径' if i == 0 else "")

# 显示广州位置
ax.scatter(guangzhou_lon, guangzhou_lat, color='green', s=100, transform=ccrs.PlateCarree(), label='出发地')
ax.text(guangzhou_lon, guangzhou_lat, '广州', verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='green', transform=ccrs.PlateCarree())

# 添加图例
plt.legend(loc='best')

plt.title('旅行路线')
plt.grid(True)
plt.show()
