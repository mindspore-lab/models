import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('merged_data_with_source.csv')

# 将中文列名转换为英文
column_mapping = {
    '名字': 'Name', '链接': 'Link', '地址': 'Address', '介绍': 'Introduction',
    '开放时间': 'OpenTime', '图片链接': 'ImageLink', '评分': 'Rating',
    '建议游玩时间': 'SuggestedPlayTime', '建议季节': 'SuggestedSeason',
    '门票': 'Ticket', '小贴士': 'Tips', 'Page': 'Page', '来源城市': 'SourceCity'
}
data.rename(columns=column_mapping, inplace=True)

# 显示清理前评分列的值分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
before_cleaning = data['Rating'].value_counts()
before_cleaning.plot(kind='bar')
plt.title('清理前')
plt.xlabel('评分')
plt.ylabel('频率')

# 删除评分列中包含空缺值、0 和 '--' 标记的行
cleanedData = data[~(data['Rating'].isna() | (data['Rating'] == 0) | (data['Rating'] == '--'))]

# 显示清理后评分列的值分布
plt.subplot(1, 2, 2)
after_cleaning = cleanedData['Rating'].value_counts()
after_cleaning.plot(kind='bar')
plt.title('清理后')
plt.xlabel('评分')
plt.ylabel('频率')

# 保存图表为图片文件
plt.tight_layout()
plt.savefig('cleaning_process.png')

# 将清理后的数据写入新的文件
cleanedData.to_csv('cleaned_data.csv', index=False)
