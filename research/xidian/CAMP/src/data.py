# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""
# import torch
# import torch
# import torchvision.transforms as transforms
import copy
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from ipdb import set_trace
import mindspore.dataset as ds
from random import shuffle, seed, choice, randint


class PrecompDataset:
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        loc = data_path + '/'

        txt_path = data_path + "/" + data_split + "_caps.txt"
        self.captions = []
        with open(txt_path, "r") as f:  # 打开文件
            for line in f.readlines():
                line = line.strip('\n')  #line是一个字符串，每一行是一个字符串 #去掉列表中每一个元素的换行符
                self.captions.append(line)

        self.images = np.load(loc + '%s_ims.npy' % data_split)

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        self.length = len(self.captions) // self.im_div


        if data_split == 'dev':
            self.length = 5000


    def __getitem__(self, index):
        # handle the image redundancy
        
        image = self.images[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        
        caption_id = index * self.im_div + randint(0, self.im_div - 1)
        caption_temp = self.captions[caption_id]
        tokens = nltk.tokenize.word_tokenize(str(caption_temp).lower().encode().decode('utf-8'))[:self.max_length]
        
        caption = []
        
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = caption

        image = np.array(image).astype(np.float32)
        target = np.array(target).astype(np.int32)
        ids = np.array([index]).astype(np.int32)
        lengths = np.array([len(caption)]).astype(np.int32)
        img_cls = copy.deepcopy(ids)
        return image, target, lengths, ids, img_cls  # , caption_mask

    def __len__(self):
        return self.length



def unbatch_concat_padded(captions):
    captions = captions[:, :87]
    return captions



def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    
    dset = PrecompDataset(data_path, data_split, vocab, opt.max_length)
    if data_split == "train":
        drop_remainder = True
    else:
        drop_remainder = False
    data_loader = ds.GeneratorDataset(dset,
                                      ["images", "captions", "lengths", "ids", "img_cls"],  #
                                      shuffle=shuffle)
    data_loader = data_loader.batch(batch_size=batch_size,
                                    pad_info={"captions": ([opt.max_length+3], 0)},
                                    drop_remainder=drop_remainder)
    return data_loader, len(dset)


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader, train_dataset_len = get_precomp_loader(dpath, 'train', vocab, opt,
                                                         batch_size, True, workers)
    val_loader, val_dataset_len = get_precomp_loader(dpath, 'dev', vocab, opt,
                                                     batch_size, False, workers)
    return train_loader, val_loader, train_dataset_len, val_dataset_len


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader, dataset_len = get_precomp_loader(dpath, split_name, vocab, opt,
                                                  batch_size, False, workers)
    return test_loader, dataset_len
