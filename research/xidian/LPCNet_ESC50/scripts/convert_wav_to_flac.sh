#!/bin/bash

# 检查是否提供了输入和输出目录
# if [ "$#" -ne 2 ]; then
#     echo "用法：\$0 <输入目录> <输出目录>"
#     exit 1
# fi

INPUT_DIR="data/esc50/train"
OUTPUT_DIR="data/esc50/train_flac"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "输入目录不存在：$INPUT_DIR"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 遍历并转换
for file in "$INPUT_DIR"/*.wav; do
    # 检查是否有 .wav 文件
    [ -e "$file" ] || { echo "没有找到 .wav 文件在 $INPUT_DIR"; exit 1; }

    filename=$(basename "$file")
    output_file="${filename%.wav}.flac"
    sox "$file" "$OUTPUT_DIR/$output_file"

    if [ $? -eq 0 ]; then
        echo "转换成功：$file -> $OUTPUT_DIR/$output_file"
    else
        echo "转换失败：$file"
    fi
done

echo "所有文件已处理完成。"


