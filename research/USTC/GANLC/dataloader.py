from ops import *
import mindspore
from mindspore.dataset import  transforms, vision
from mindspore.dataset import GeneratorDataset
import cv2 as cv
import collections, os, math
import numpy as np
from scipy import signal
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.io import loadmat
import glob
import  random


class train_dataset:

    def __init__(self,args):
        root=args.input_medical_dir
        mode=args.mode
        self.rnn_num=args.RNN_N
        if mode=='inference':
            mode='test'
        self.aligned=True
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/Sinogram' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/FBP' % mode) + '/*.*'))

        self.files_B = sorted(glob.glob(os.path.join(root, '%s/Phantom' % mode) + '/*.*'))


    def __getitem__(self, index):
        start_for_next = [224, 544, 778, 1017, 1227, 1570, 1784, 2028, 2238]
        Sinogram,Fbp,Phantom=[],[],[]
        for i in range(9):
            if (index<start_for_next[i]) &(index>start_for_next[i]-2):
                index=start_for_next[i]-2
                break
        for i in range(self.rnn_num):
            sinogram = mindspore.tensor(np.load(self.files_A[(index + i) % len(self.files_A)]))
            fbp = mindspore.tensor(np.load(self.files_C[(index + i) % len(self.files_C)]))
            phantom = mindspore.tensor(np.load(self.files_B[(index + i) % len(self.files_B)]))
            Sinogram.append(sinogram.unsqueeze(0).unsqueeze(0))
            Fbp.append(fbp.unsqueeze(0).unsqueeze(0))
            Phantom.append(phantom.unsqueeze(0).unsqueeze(0))
        Sinogram = mindspore.ops.cat(Sinogram, axis=0)
        Fbp = mindspore.ops.cat(Fbp, axis=0)
        Phantom = mindspore.ops.cat(Phantom, axis=0)
        return [ Fbp.float(),Phantom.float(),Sinogram.float()]


    def __len__(self):
        return max([len(self.files_A), len(self.files_B), len(self.files_C)])


