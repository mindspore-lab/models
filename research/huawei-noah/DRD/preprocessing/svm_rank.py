# -*- coding: utf-8 -*-
import os,sys
import time
import subprocess
# Windows .exe
# SVMRANK_TRAIN = r'start ./preProcessing/svm_rank_learn.exe'
# SVMRANK_TEST = r'start ./preProcessing/svm_rank_classify.exe'
# Linux
SVMRANK_TRAIN = './preprocessing/svm_rank_learn'
SVMRANK_TEST = './preprocessing/svm_rank_classify'

def svm_rank(dataset_path, c = 20):
	train_file = dataset_path + 'cleaned/' + 'train.txt'
	test_file = dataset_path + 'cleaned/' + 'test.txt'
	vali_file = dataset_path + 'cleaned/' + 'vali.txt'
	sample_file = dataset_path + 'cleaned/' + 'sample.txt'

	dataset_path = dataset_path + 'predict/'
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)
	model_file = dataset_path + 'svmRank.model'

	# train a svmRank as production ranker
	train_command = SVMRANK_TRAIN + ' -c ' + str(c) + ' ' + sample_file + ' ' + model_file
	print(train_command)
	# Windows
	# child = subprocess.Popen(["cmd", "/c", train_command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# Linux
	child = subprocess.Popen(["/bin/sh", "-c", train_command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# print(child.stdout.readlines()[0].decode("gbk"))
	child.wait()



	# use trained svmRank to get initialized rank prediction
	command = SVMRANK_TEST + ' ' + train_file + ' ' + model_file + ' ' + dataset_path + 'train_predict.txt'
	print(command)
	# Windows
	# child = subprocess.Popen(["cmd", "/c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# Linux
	child = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# print(child.stdout.readlines())
	child.wait()


	command = SVMRANK_TEST + ' ' + vali_file + ' ' + model_file + ' ' + dataset_path + 'vali_predict.txt'
	print(command)
	# child = subprocess.Popen(["cmd", "/c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	child = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# print(child.stdout.readlines())
	child.wait()


	command = SVMRANK_TEST + ' ' + test_file + ' ' + model_file + ' ' + dataset_path + 'test_predict.txt'
	print(command)
	# child = subprocess.Popen(["cmd", "/c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	child = subprocess.Popen(["/bin/sh", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	# print(child.stdout.readlines())
	child.wait()



if __name__ == "__main__":
    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]
        svm_rank(DATASET_PATH)
    else:
        DATASET_PATH = sys.argv[1]
        C = sys.argv[2]
        svm_rank(DATASET_PATH,C)