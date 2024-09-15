import pandas as pd

# 读取文件
output_file="/home/xidian/btq/lsy/models/research/recommend/ncf/data/amazon_video/ratings.csv"
df = pd.read_csv(output_file, names=['user_id', 'item_id', 'rating', 'timestamp'])

# 计算每个用户有多少个不同的item_id
user_item_counts = df.groupby('user_id')['item_id'].nunique()
# 过滤掉拥有少于20个item_id的用户
filtered_users = user_item_counts[user_item_counts >= 20].index
# 保留这些用户的数据
df_filtered = df[df['user_id'].isin(filtered_users)]
# 将user_id重新从1开始排序
user_map = {old_id: new_id for new_id, old_id in enumerate(df_filtered['user_id'].unique(), start=1)}
df_filtered['user_id'] = df_filtered['user_id'].map(user_map)
item_map = {old_id: new_id for new_id, old_id in enumerate(df_filtered['item_id'].unique(), start=1)}
df_filtered['item_id'] = df_filtered['item_id'].map(item_map)
df_filtered = df_filtered.sort_values(by='user_id')
# 将处理后的数据保存到一个新的CSV文件中
df_filtered.to_csv(output_file, index=False)

print(f"处理完成，共保留 {len(df_filtered)} 行数据，用户ID已重新排序。")
