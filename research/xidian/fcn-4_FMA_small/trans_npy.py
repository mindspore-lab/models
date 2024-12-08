import os
import numpy as np
import librosa


# from scipy.io import wavfile

# def convert_wav_to_npy(wav_file_path, npy_file_path):
#     sample_rate, audio_data = wavfile.read(wav_file_path)
#     np.save(npy_file_path, audio_data)

def convert_wav_to_npy(wav_file_path, npy_file_path):
    # 使用 librosa 加载音频文件
    audio_data, sample_rate = librosa.load(wav_file_path, sr=None)
    # 保存为 .npy 文件
    np.save(npy_file_path, audio_data)

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                npy_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npy')
                convert_wav_to_npy(wav_path, npy_path)
                print(f'Converted {wav_path} to {npy_path}')

# 使用示例
input_directory = 'data/genres_original/'
output_directory = 'data/npy/'

process_directory(input_directory, output_directory)
