#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "用法：\$0 <输入目录> <输出目录>"
    exit 1
fi

INPUT_DIR=" " # .wav file dir
OUTPUT_DIR=" " # .flac file dir

if [ ! -d "$INPUT_DIR" ]; then
    echo "input dir no exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.wav; do
    [ -e "$file" ] || { echo "no .wav files in $INPUT_DIR"; exit 1; }

    filename=$(basename "$file")
    output_file="${filename%.wav}.flac"
    sox "$file" "$OUTPUT_DIR/$output_file"

    if [ $? -eq 0 ]; then
        echo "sucess: $file -> $OUTPUT_DIR/$output_file"
    else
        echo "false: $file"
    fi
done

echo "finished"