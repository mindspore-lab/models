#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import mindspore as ms
import mindspore.dataset as data
from ipdb import set_trace

from functools import reduce
from settings import input_json, input_label_h5, input_att_dir, input_fc_dir, input_attr_dir
class DataLoader(object):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img  
        # self.sampler = data.Sampler()
        self.use_att = getattr(opt, 'use_att', True)  
        self.use_box = getattr(opt, 'use_box', True)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)  
        print('use_att:',self.use_att)
        print('DataLoader loading json file: ', input_json)
        self.info = json.load(open(input_json))
        self.ix_to_word = self.info['ix_to_word']    
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        print('DataLoader loading h5 file: ', input_fc_dir, input_att_dir,input_label_h5)
        self.h5_label_file = h5py.File(input_label_h5, 'r', driver='core')

        self.input_fc_dir = input_fc_dir
        self.input_att_dir = input_att_dir
        self.input_box_dir = ""
        self.input_attr_dir = input_attr_dir

        seq_size = self.h5_label_file['labels'].shape  
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length) 
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]   
        self.num_images = self.label_start_ix.shape[0]  
        print('read %d image features' %(self.num_images))

        self.split_ix = {'train': [], 'val': [], 'eval': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['eval'].append(ix)
            elif opt.train_only == 0: 
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train'])) 
        print('assigned %d images to split val' %len(self.split_ix['val'])) 
        print('assigned %d images to split test' %len(self.split_ix['eval'])) 

        self.iterators = {'train': 0, 'val': 0, 'eval': 0}
        
        self._prefetch_process = {} 
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # set_trace()
        ix1 = self.label_start_ix[ix] - 1 
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
        ixl = random.randint(ix1, ix2 - seq_per_img + 1)
        seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] 
        att_batch = [] 
        attr_batch = [] 
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):  
            # set_trace()
            tmp_dict = self._prefetch_process[split].get()
            # set_trace()
            tmp_fc = tmp_dict["fc_feats"][0]
            tmp_att = tmp_dict["att_feats"][0]
            tmp_attr = tmp_dict["attr_feats"][0]
            ix = int(tmp_dict["ix"])
            # print("ix:",ix)
            # print("fc_feats:",tmp_att)
            tmp_wrapped = tmp_dict["wrapped"]
            fc_batch.append(tmp_fc)     
            att_batch.append(tmp_att)
            attr_batch.append(tmp_attr)
            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)
            if tmp_wrapped:
                wrapped = True

            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
        # set_trace()
        fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*seq_per_img for _ in fc_batch]))
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = 1

        max_attr_len = max([_.shape[0] for _ in attr_batch])
        data['attr_feats'] = np.zeros([len(attr_batch)*seq_per_img,max_attr_len,attr_batch[0].shape[1]],dtype = "float32")
        for i in range(len(attr_batch)):
            data['attr_feats'][i*seq_per_img:(i+1)*seq_per_img, :attr_batch[i].shape[0]] = attr_batch[i]

        data['labels'] = np.vstack(label_batch)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos
        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index   
        # if ix==100:
        #     set_trace()   
        # print("feat_ix:",ix)  
        # print("id:",self.info['images'][ix]['id'])
        att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        attr_feat = np.load(os.path.join(self.input_attr_dir, str(self.info['images'][ix]['id']) + '.npy'))
        attr_feat = attr_feat.reshape(-1, attr_feat.shape[-1])
        return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
                att_feat,
                attr_feat,
                ix)

    def __len__(self):
        return len(self.info['images'])

class BlobFetcher():
    def __init__(self, split, dataloader, if_shuffle=False):
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    def reset(self):
        # set_trace()
        new_sampler = data.SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:])
        dataset = data.GeneratorDataset(source=self.dataloader,column_names=["fc_feats", "att_feats","attr_feats","ix"], sampler=new_sampler)
        dataset = dataset.batch(1)
        self.split_loader = iter(dataset.create_dict_iterator(output_numpy=True))


    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        # set_trace()
        # print(ix)
        tmp = next(self.split_loader)
        if wrapped:
            self.reset()
        assert int(tmp["ix"]) == ix, "ix not equal"
        tmp["wrapped"] = wrapped
        return tmp
