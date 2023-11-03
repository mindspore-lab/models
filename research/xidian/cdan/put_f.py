import os
import numpy as np
import os
import shutil
import time

file_data = ""
data_list = []
data_label = []
with open("../data/svhn2mnist/svhn_balanced.txt", "r") as f:
    for line in f:
        line = line[:-1]  # 去掉换行符(方法1）
        a = line.split(' ')
        # print(a)  # 截取倒数第三位到结尾
        data_list.append(a[0])
        data_label.append("../"+a[1])

print(max(data_label))
print(len(data_label))


for i in range(len(data_label)):
    if data_label[i] == '../0':
        continue
    if data_label[i] == '../1':
        continue
    if data_label[i] == '../2':
        continue
    if data_label[i] == '../3':
        continue
    if data_label[i] == '../4':
        continue
    if data_label[i] == '../5':
        continue
    if data_label[i] == '../6':
        continue
    if data_label[i] == '../7':
        continue
    if data_label[i] == '../8':
        continue

    shutil.move(data_list[i], data_label[i])
    #time.sleep(0.5)
    print('\r', "{:d}".format(len(data_label) - i), end='', flush=True)

print('\n1')
# with open("mnist_test.txt", "w") as f:
#     f.write(file_data)



# # 读取txt文件并将其转化为array
# f = open(r"svhn_balanced.txt")
# line = f.readline()
# data_list = []
# while line:
#     num = list(map(float, line.split(' ')))
#     data_list.append(num)
#     line = f.readline()
# f.close()
# data_array = np.array(data_list)
# # print(data_array[:,0])
#
# # 读取每张图片按照其分类复制到相应的文件夹中
# imgs = os.listdir('./test')
# imgnum = len(imgs)  # 文件夹中图片的数量
# j = 1
# for i in range(imgnum):  # 遍历每张图片
#     # print(int(data_array[i][0]))
#     label=int(data_array[i][0])    #图片对应的类别
#     shutil.move('./test/'+str(j)+'.jpg', './'+str(label)+'/'+str(j)+'.jpg')
#     j+=1



