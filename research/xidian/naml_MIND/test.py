import pickle

# 读取 .pkl 文件
with open(r'D:\file\Project_py\models-master\research\recommend\naml\MINDlarge\MINDlarge_utils\uid2index.pkl', 'rb') as file:
    data = pickle.load(file)

# 查看数据
print(data)