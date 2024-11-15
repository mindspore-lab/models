#!/bin/bash

# 检查参数数量
if [ $# != 2 ]; then
    echo "Usage: bash convert_mp3_to_flac.sh [SOURCE_DIRECTORY] [DESTINATION_DIRECTORY]"
    exit 1
fi

# 获取源目录和目标目录
source_dir=$(realpath $1)
dest_dir=$(realpath $2)

# 检查源目录是否存在
if [ ! -d "$source_dir" ]; then
    echo "error: SOURCE_DIRECTORY=$source_dir does not exist"
    exit 1
fi

# 创建目标目录（如果不存在）
if [ ! -d "$dest_dir" ]; then
    mkdir -p "$dest_dir"
fi

# 创建日志文件
log_file="$dest_dir/conversion_log.txt"
echo "Conversion log:" > "$log_file"

# 遍历源目录中的所有 .mp3 文件
for mp3_file in "$source_dir"/*.mp3; do
    if [ -f "$mp3_file" ]; then
        # 获取文件名（不包括扩展名）
        base_name=$(basename "$mp3_file" .mp3)

        # 构建目标文件路径
        flac_file="$dest_dir/$base_name.flac"

        # 使用 ffmpeg 转换文件
        ffmpeg -i "$mp3_file" -c:a flac "$flac_file" >> "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "Converted $mp3_file to $flac_file"
            echo "Converted $mp3_file to $flac_file" >> "$log_file"
        else
            echo "Failed to convert $mp3_file"
            echo "Failed to convert $mp3_file" >> "$log_file"
        fi
    fi
done

echo "Conversion complete. Check $log_file for details."