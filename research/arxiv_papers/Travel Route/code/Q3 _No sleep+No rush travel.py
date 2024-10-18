import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import nn, Tensor
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

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
    R = ms.Tensor(6371, ms.float32)  # 地球半径(km)
    dLat = ops.deg2rad(lat2 - lat1)
    dLon = ops.deg2rad(lon2 - lon1)
    a = ops.sin(dLat/2)**2 + ops.cos(ops.deg2rad(lat1)) * ops.cos(ops.deg2rad(lat2)) * ops.sin(dLon/2)**2
    c = 2 * ops.asin(ops.sqrt(a))
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
                city_lat, city_lon = guangzhou_lat, guangzhou_lon
            else:
                current_city_data = city_data[city_data['城市'] == current_city].iloc[0]
                city_lat, city_lon = current_city_data['纬度'], current_city_data['经度']
            
            target_lat, target_lon = city_row['纬度'], city_row['经度']
            
            # 计算距离，转换为公里
            dist = haversine_distance(ms.Tensor(city_lat, ms.float32), ms.Tensor(city_lon, ms.float32), 
                                      ms.Tensor(target_lat, ms.float32), ms.Tensor(target_lon, ms.float32))
            travel_time = dist / high_speed  # 转换为小时
            travel_cost = dist * cost_per_km  # 元

            spot_data = scenic_data[scenic_data['来源城市'] == city]
            if spot_data.empty:
                continue
            # 选择评分最高的景点
            best_spot = spot_data.loc[spot_data['评分'].idxmax()]
            visit_time = ms.Tensor(best_spot['建议游玩时间'], ms.float32)  # 使用小时
            total_city_time = travel_time + visit_time
            
            # 调试信息
            print(f'评估城市: {city}，旅行时间: {travel_time:.2f}小时，游玩时间: {visit_time:.2f}小时，总时间: {total_city_time:.2f}小时')
            
            if current_time + total_city_time <= total_time_limit:
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
    current_time += best_city_time
    current_cost += best_city_cost
    total_spots += ms.Tensor(1, ms.int32)
    current_city = best_city
    
    # 调试信息
    print(f'访问城市: {best_city}，当前总时间: {current_time:.2f}小时，当前总费用: {current_cost:.2f}元')

# 输出结果
print(f'Travel Route: {" -> ".join(visited_cities)}')
print(f'Total Travel Time (hours): {current_time:.2f}')
print(f'Total Cost: {current_cost:.2f}')
print(f'Total Scenic Spots: {total_spots}')

# 可视化结果
plt.figure(figsize=(12, 8))
m = Basemap(llcrnrlon=70, llcrnrlat=15, urcrnrlon=140, urcrnrlat=55,
            projection='lcc', lat_1=33, lat_2=45, lon_0=100)

m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

for i, city in enumerate(visited_cities):
    city_data_row = city_data[city_data['城市'] == city].iloc[0]
    x, y = m(city_data_row['经度'], city_data_row['纬度'])
    m.plot(x, y, 'ro', markersize=10)
    plt.text(x, y, f'{i+1}. {city}', fontsize=8, ha='right', va='bottom')

for i in range(len(visited_cities) - 1):
    start = city_data[city_data['城市'] == visited_cities[i]].iloc[0]
    end = city_data[city_data['城市'] == visited_cities[i+1]].iloc[0]
    m.drawgreatcircle(start['经度'], start['纬度'], end['经度'], end['纬度'], linewidth=1, color='b')

plt.title('旅行路线')
plt.show()

# 显示总花费时间、总费用和游玩景点数量
print(f'Total Travel Time (minutes): {current_time*60:.0f}')
print(f'Total Cost: {current_cost:.2f}')
print(f'Total Scenic Spots: {total_spots}')
