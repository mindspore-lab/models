import os
import csv

# 定义文件夹路径和输出文件路径
folder_path = '附件'  # 你的文件夹路径
output_file = 'merged_data_with_source.csv'  # 合并后的输出文件

# 获取文件夹中的所有CSV文件
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 打开输出文件
with open(output_file, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)

    # 读取第一个文件并写入表头到输出文件，并增加一列用于区分来源
    with open(os.path.join(folder_path, files[0]), 'r', encoding='utf-8') as fin:
        header = next(csv.reader(fin))
        writer.writerow(header + ['来源城市'])

    # 遍历所有文件并合并内容
    for file in files:
        # 构建文件路径
        file_path = os.path.join(folder_path, file)
        
        # 提取文件名（不包括扩展名）作为城市名
        city_name = os.path.splitext(file)[0]
        
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin)
            next(reader)  # 跳过表头
            
            # 读取文件内容并写入输出文件，同时添加城市名列
            for row in reader:
                writer.writerow(row + [city_name])

print('文件合并完成。')
