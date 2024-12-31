from mindspore.mindrecord import FileReader

# 指定 MindRecord 文件路径
# file_name = "/data1/wkr/data/kkbox_mindrecord/train_input_part.mindrecord01"
file_name = "/data1/wkr/data/avazu_mindrecord/train_input_part.mindrecord01"

# 创建文件读取器
reader = FileReader(file_name)

# 遍历文件中的每一条数据
for i, data in enumerate(reader.get_next()):
    print(f"样本 {i + 1}: {data}")

    # 如果样本太多，可以限制打印数量
    if i >= 10:  # 仅打印前 10 条数据
        break

# 关闭读取器
reader.close()
