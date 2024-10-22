import random
import gc

def RandomShuffle(infile, outfile, deleteSchema=False):
	with open(infile, 'r', encoding='utf-8') as fs:
		arr = fs.readlines()
	if not arr[-1].endswith('\n'):
		arr[-1] += '\n'
	if deleteSchema:
		arr = arr[1:]
	random.shuffle(arr)
	with open(outfile, 'w', encoding='utf-8') as fs:
		for line in arr:
			fs.write(line)
	del arr

def WriteToBuff(buff, line, out):
	BUFF_SIZE = 1000000
	buff.append(line)
	if len(buff) == BUFF_SIZE:
		WriteToDisk(buff, out)

def WriteToDisk(buff, out):
	with open(out, 'a', encoding='utf-8') as fs:
		for line in buff:
			fs.write(line)
	buff.clear()


def SubDataSet(infile, outfile1, outfile2, rate):
	out1 = list()
	out2 = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			if random.random() < rate:
				WriteToBuff(out1, line, outfile1)
			else:
				WriteToBuff(out2, line, outfile2)
	WriteToDisk(out1, outfile1)
	WriteToDisk(out2, outfile2)

def CombineFiles(files, out):
	buff = list()
	for file in files:
		with open(file, 'r') as fs:
			for line in fs:
				WriteToBuff(buff, line, out)
	WriteToDisk(buff, out)