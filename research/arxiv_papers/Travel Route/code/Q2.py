import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import nn, Tensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

# 读取数据
data = pd.read_excel('问题二数据集.xlsx')

# 数据清洗和预处理
data = data.dropna()  # 删除缺失值

# 分类
city_scale = data[['线路密度 (km/km²)', '高速公路里程 (km)', '机场航班数量']]
environment = data[['AQI', '绿化覆盖率 (%)', '废水处理率 (%)', '废气处理率 (%)', '垃圾分类处理率 (%)']]
culture = data[['历史遗迹数量', '博物馆数量', '文化活动频次', '文化设施数量']]
transport = data[['公共交通覆盖率 (%)', '线路密度 (km/km²)']]
climate = data[['年平均气温 (℃)', '年降水量 (mm)', '适宜旅游天数', '空气湿度 (%)']]
food = data[['餐馆数量', '特色美食数量', '美食活动频次']]

categories = [city_scale, environment, culture, transport, climate, food]
category_names = ['City Scale', 'Environment', 'Culture', 'Transport', 'Climate', 'Food']
scores = mnp.zeros((len(data), len(categories)))

def kmo_test(data):
    return calculate_kmo(data)[1]

class StandardScaler(nn.Cell):
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def construct(self, x):
        if self.mean is None:
            self.mean = mnp.mean(x, axis=0)
            self.std = mnp.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

class PCA(nn.Cell):
    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components

    def construct(self, x):
        cov = mnp.cov(x.T)
        eigenvalues, eigenvectors = mnp.linalg.eig(cov)
        idx = mnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        return mnp.dot(x, eigenvectors[:, :self.n_components])

def topsis(data):
    scaler = StandardScaler()
    normalized_data = scaler(Tensor(data.values, dtype=ms.float32)).asnumpy()
    entropy = -mnp.sum(normalized_data * mnp.log(normalized_data + 1e-10), axis=0) / mnp.log(len(normalized_data))
    weights = (1 - entropy) / mnp.sum(1 - entropy)
    ideal_solution = mnp.max(normalized_data, axis=0)
    negative_solution = mnp.min(normalized_data, axis=0)
    distance_to_ideal = mnp.sqrt(mnp.sum((normalized_data - ideal_solution)**2, axis=1))
    distance_to_negative = mnp.sqrt(mnp.sum((normalized_data - negative_solution)**2, axis=1))
    return distance_to_negative / (distance_to_ideal + distance_to_negative)

for i, category in enumerate(categories):
    kmo_value = kmo_test(category)
    
    if kmo_value > 0.6:  # 假设通过KMO检验的阈值为0.6
        pca = PCA(n_components=1)
        scores[:, i] = pca(Tensor(category.values, dtype=ms.float32)).asnumpy().flatten()
    else:
        scores[:, i] = topsis(category)
    
    # 可视化降维结果
    plt.figure()
    plt.bar(range(len(data)), scores[:, i])
    plt.title(f"{'PCA' if kmo_value > 0.6 else 'TOPSIS'} dimensionality reduction results - {category_names[i]}")
    plt.xlabel('City Index')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(f'dimensionality_reduction_{category_names[i]}.png')
    plt.close()

# 综合评价模型构建（熵权法的TOPSIS）
topsis_score = topsis(pd.DataFrame(scores))

# 选出前50个城市
sorted_index = mnp.argsort(topsis_score)[::-1]
top50_cities = data.iloc[sorted_index[:50].asnumpy()]

# 可视化结果
plt.figure(figsize=(15, 8))
plt.bar(range(50), topsis_score[sorted_index[:50]])
plt.title('Top 50 Cities Most Attractive to Foreign Tourists')
plt.xlabel('City')
plt.ylabel('TOPSIS Score')
plt.xticks(range(50), top50_cities['来源城市'], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig('top50_cities.png')
plt.close()

# 保存结果
top50_cities.to_csv('top50_cities.csv', index=False)

# 输出排名结果为表格文件
full_ranking = data.iloc[sorted_index.asnumpy()]
full_ranking.to_csv('full_city_ranking.csv', index=False)
