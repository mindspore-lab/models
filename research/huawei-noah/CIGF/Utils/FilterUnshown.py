import numpy as np
import pickle
from Params import *
import MakeData
from scipy.sparse import csr_matrix
from ToolScripts.TimeLogger import log

def shrink(train, cv, test):
	row = train.shape[0]
	nums = np.sum(train + cv + test, axis=1)
	locs = list(map(lambda x: x[0], np.argwhere(nums>0)))
	outTrain = train[locs]
	outTest = test[locs]
	outCv = cv[locs]
	return outTrain, outCv, outTest

def transpose(train, cv, test):
	return np.transpose(train), np.transpose(cv), np.transpose(test)

maker = MakeData.ScipyMatMaker()
train = maker.ReadMat(TRAIN_FILE)
cv = maker.ReadMat(CV_FILE)
test = maker.ReadMat(TEST_FILE)
print(train.shape)
train, cv, test = shrink(train, cv, test)
train, cv, test = transpose(train, cv, test)
train, cv, test = shrink(train, cv, test)
train, cv, test = transpose(train, cv, test)
# with open(TRAIN_FILE, 'wb') as fs:
# 	pickle.dump(train, fs)
# with open(TEST_FILE, 'wb') as fs:
# 	pickle.dump(test, fs)
# with open(CV_FILE, 'wb') as fs:
# 	pickle.dump(cv, fs)
print(train.shape)