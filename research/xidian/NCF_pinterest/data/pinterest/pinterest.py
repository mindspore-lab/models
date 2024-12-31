import pandas as pd
# https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data
# 读取 Pinterest 数据集
input_file = "/home/xidian/btq/lsy/models/research/recommend/ncf/data/pinterest/pinterest.rating"
output_file = "/home/xidian/btq/lsy/models/research/recommend/ncf/data/pinterest/rating.csv"

# 读取数据
df = pd.read_csv(input_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
df.to_csv(output_file, index=False)
df['user_id'] = df['user_id'].astype('int64')
df['item_id'] = df['item_id'].astype('int64')
df['rating'] = df['rating'].astype('int64')
df['timestamp'] = df['timestamp'].astype('int64')
df = df[df['user_id'] != 0]
df.to_csv(output_file, index=False)
print(f"Data successfully converted and saved to {output_file}")
print(df.dtypes)
df = pd.read_csv(output_file)
# 去掉重复行
df_cleaned = df.drop_duplicates()
# 保存到新的CSV文件
df_cleaned.to_csv(output_file, index=False)
print("Duplicates removed and saved to 'ratings_cleaned.csv'.")

df = pd.read_csv(output_file)
# 计算每个用户的item_id数量
user_item_counts = df.groupby('user_id')['item_id'].count()
# 找出item_id数量小于20的用户
users_to_remove = user_item_counts[user_item_counts < 20].index
# 去除这些用户的所有行
df_filtered = df[~df['user_id'].isin(users_to_remove)]
# 重新排序用户ID，使得ID从1开始
df_filtered['user_id'] = pd.factorize(df_filtered['user_id'])[0] + 1
# 保存处理后的数据到新的CSV文件
df_filtered.to_csv(output_file, index=False)
print("Filtered data saved to 'ratings_filtered.csv'.")