import glob
import os
from ipdb import set_trace

##将flac转化为wav
def flac_to_wav(filepath, savedir):
    filename = savedir
    savefilename = filename.split('/')
    save_dir = savedir
    print(save_dir)
    cmd = 'sox ' + filepath + ' ' + save_dir
    os.system(cmd)

##读取txt文件
def open_text(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        return lines
    
##处理txt文件
def function_txt(label, path):
    label_id = label.split(' ')[0]
    label = label[len(label_id)+1:]
    path = path.replace(path.split('/')[-1], label_id + '.txt')
    with open(path, 'w') as f:
        f.write(label)

path1 = "/media/data1/xidian/wk/ctcmodel/data/TIMIT/data/lisa/data/timit/raw/Libri_mini/*/*/*/*.flac"
wav_file1 = glob.glob(path1)
path2 = "/media/data1/xidian/wk/ctcmodel/data/TIMIT/data/lisa/data/timit/raw/Libri_mini/*/*/*/*.trans.txt"
wav_file2 = glob.glob(path2)

for audio_path in wav_file1:
    #生成wav文件
    savedir = audio_path.replace('.flac', '.wav')
    flac_to_wav(audio_path, savedir)

for file_path in wav_file2:
    #生成text文件
    labels = open_text(file_path)
    for label in labels:
        function_txt(label, file_path)

path3 = "/media/data1/xidian/wk/ctcmodel/data/TIMIT/data/lisa/data/timit/raw/Libri_mini/*/*/*/*.txt"
wav_file3 = glob.glob(path2)
for file_path in wav_file2:
    with open(file_path, 'r+') as file:
        # 读取文件内容
        file_content = file.read()
        
        # 将大写字母转换为小写字母
        modified_content = file_content.lower()
        
        # 将文件指针移动到文件开头并覆盖原文件内容
        file.seek(0)
        file.write(modified_content)
        file.truncate()
