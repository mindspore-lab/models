import os
from pydub import AudioSegment

# 定义源目录和目标目录
source_dir = "si_lk"
output_dir = "converted_flac_files"

# 创建目标目录
os.makedirs(output_dir, exist_ok=True)

# 遍历 si_lk 文件夹中的每个 wav 文件
for filename in os.listdir(source_dir):
    if filename.endswith(".wav"):
        # 定义文件的完整路径
        wav_path = os.path.join(source_dir, filename)
        
        # 将文件名的扩展名从 .wav 替换为 .flac
        flac_filename = os.path.splitext(filename)[0] + ".flac"
        
        # 读取 wav 文件并转换为 flac
        audio = AudioSegment.from_wav(wav_path)
        
        # 创建嵌套文件夹结构
        outer_folder = os.path.join(output_dir, os.path.splitext(filename)[0])
        inner_folder = os.path.join(outer_folder, "inner_folder")
        os.makedirs(inner_folder, exist_ok=True)
        
        # 保存转换后的 flac 文件到目标路径
        flac_path = os.path.join(inner_folder, flac_filename)
        audio.export(flac_path, format="flac")
        
        print(f"转换完成: {wav_path} -> {flac_path}")

