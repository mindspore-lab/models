# https://aistudio.baidu.com/datasetdetail/109008

import pandas as pd

# 读取数据
df = pd.read_csv(r'/home/xidian/btq/lsy/models/research/recommend/ncf/data/douban/ratings.csv', sep=',', header=0)
user_item_counts = df.groupby('USER_MD5')['MOVIE_ID'].nunique()

# 找出拥有少于 20 个 item_id 的用户
users_to_keep = user_item_counts[user_item_counts >= 20].index

# 删除拥有少于 20 个 item_id 的用户对应的行
df = df[df['USER_MD5'].isin(users_to_keep)]
# 重新排列用户ID
user_map = {user: idx + 1 for idx, user in enumerate(df['USER_MD5'].unique())}
df['USER_MD5'] = df['USER_MD5'].map(user_map)
# 将RATING_TIME一列置为0
df['RATING_TIME'] = 0

# 删去RATING_ID这一列
df = df.drop(columns=['RATING_ID'])
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
# 保存修改后的文件
unique_items = df['item_id'].unique()  # 获取唯一的 item_id 列表
item_id_map = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 生成 item_id 的新编号，从1开始
# 将原有的 item_id 替换为新的编号
df['item_id'] = df['item_id'].map(item_id_map)
# 检查替换后的结果
print(df.head())
df = df.sort_values(by='user_id')
df.to_csv(r'/home/xidian/btq/lsy/models/research/recommend/ncf/data/douban/ratings.csv', index=False)
