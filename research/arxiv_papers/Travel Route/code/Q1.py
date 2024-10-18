import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv('cleaned_data.csv')

# 将中文列名转换为英文（如果还没有转换）
column_mapping = {
    '名字': 'Name', '链接': 'Link', '地址': 'Address', '介绍': 'Introduction',
    '开放时间': 'OpenTime', '图片链接': 'ImageLink', '评分': 'Rating',
    '建议游玩时间': 'SuggestedPlayTime', '建议季节': 'SuggestedSeason',
    '门票': 'Ticket', '小贴士': 'Tips', 'Page': 'Page', '来源城市': 'SourceCity'
}
data.rename(columns=column_mapping, inplace=True)

# 找出最高评分
max_rating = data['Rating'].max()

# 统计每个城市获得最高评分的景点数量
max_rating_data = data[data['Rating'] == max_rating]
city_counts = max_rating_data['SourceCity'].value_counts().reset_index()
city_counts.columns = ['SourceCity', 'GroupCount']

# 提取前10个城市
top10_cities = city_counts.head(10)

# 可视化前10个城市的结果
plt.figure(figsize=(12, 6))
sns.barplot(x='SourceCity', y='GroupCount', data=top10_cities, palette='husl')

# 设置图表标题和轴标签
plt.title('Top 10 Cities with the Most Attractions Having the Highest Score', fontsize=14)
plt.xlabel('City', fontsize=12)
plt.ylabel('Number of Attractions with the Highest Score', fontsize=12)

# 设置X轴刻度标签
plt.xticks(rotation=45, ha='right')

# 添加图例
plt.legend(['Number of Attractions'], loc='upper right')

# 设置背景颜色
plt.gca().set_facecolor('#E6E6E6')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 显示最高评分和全国获评这个最高评分的景点数量
print(f"The highest score (BS) is: {max_rating}")
print(f"The number of attractions with the highest score (BS) is: {len(max_rating_data)}")

# 保存图表为图片文件
plt.tight_layout()
plt.savefig('top10_cities_highest_score.png')

# 保存前10个城市的表格到文件
top10_cities.to_csv('top10_cities_highest_score.csv', index=False)

# 显示图表
plt.show()
