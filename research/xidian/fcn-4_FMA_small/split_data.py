import os
import csv
import random

def generate_csv_and_split(data_dir, categories, output_csv_a, output_csv_b, split_ratio=0.8):
    data_list = []

    # 遍历每个类别目录
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # 遍历类别目录中的每个音频文件
        for filename in os.listdir(category_dir):
            if filename.endswith('.wav'):
                # file_path = os.path.join(category_dir, filename)
                file_path = os.path.join(category, filename)
                # 创建类别向量
                category_vector = [1 if category == cat else 0 for cat in categories]
                # 将类别向量和文件路径添加到列表中
                data_list.append(category_vector + [file_path])

    # 随机打乱列表
    random.shuffle(data_list)

    # 划分数据
    split_index = int(len(data_list) * split_ratio)
    data_a = data_list[:split_index]
    data_b = data_list[split_index:]

    # 写入CSV文件A
    with open(output_csv_a, 'w', newline='') as csvfile_a:
        writer_a = csv.writer(csvfile_a)
        writer_a.writerows(data_a)

    # 写入CSV文件B
    with open(output_csv_b, 'w', newline='') as csvfile_b:
        writer_b = csv.writer(csvfile_b)
        writer_b.writerows(data_b)

# 使用示例
data_directory = 'data/genres_original/'
categories = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
output_csv_a = 'data/music_tagging_train_tmp.csv'
output_csv_b = 'data/music_tagging_val_tmp.csv'

generate_csv_and_split(data_directory, categories, output_csv_a, output_csv_b)
