with open('./data_stego/7b-48.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

from collections import OrderedDict
lines_unique = list(OrderedDict.fromkeys(lines))

with open('./data_stego/7b-48.txt', 'w', encoding='utf-8') as file:
    file.writelines(lines_unique)
