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
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from ipdb import set_trace
import mindspore.dataset as ds


# import torch.utils.data as data

class PrecompDataset:
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'


        self.captions = []
        txt_path = data_path + "/" + data_split + "_caps.txt"
        self.captions = []
        with open(txt_path, "r") as f:  # 打开文件
            for line in f.readlines():
                line = line.strip('\n')  #line是一个字符串，每一行是一个字符串 #去掉列表中每一个元素的换行符
                self.captions.append(line)

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length =  len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for cocos is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = self.images[img_id]
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower().encode().decode('utf-8'))

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = caption
        caption_mask = np.zeros((87))
        caption_mask[:len(caption)] += 1 

        image = np.array(image).astype(np.float32)
        target = np.array(target).astype(np.int32)
        ids = np.array([index]).astype(np.int32)
        lengths = np.array([len(caption)]).astype(np.int32)
        caption_mask = caption_mask.astype(np.int32)
        return image,  target, lengths, ids, caption_mask

    def __len__(self):
        return self.length
#



def unbatch_concat_padded(captions):

    captions = captions[:,:87]
    return captions




def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)
    # set_trace()
    if data_split == "train":
        drop_remainder = True
    else:
        drop_remainder = False
    data_loader = ds.GeneratorDataset(dset,
                                      ["images", "captions","lengths", "ids", "caption_mask"],   #
                                      shuffle=shuffle)
    data_loader = data_loader.batch(batch_size = batch_size,
                      pad_info={"captions":([87], 0)},
                      drop_remainder = drop_remainder)

    return data_loader,len(dset)


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader,train_dataset_len = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader, val_dataset_len = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader,train_dataset_len,val_dataset_len


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader,dataset_len = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader,dataset_len
