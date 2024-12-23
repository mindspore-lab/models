from __future__ import print_function, absolute_import

import math
import random

import h5py
import numpy as np
from PIL import Image
from mindspore.dataset import Dataset
from mindspore import Tensor, ops
import mindspore
train_hdf_ = '../VCM-POSE/dataset-train.h5'
test_hdf_ = '../VCM-POSE/dataset-test.h5'


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)

        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        hdf5_file = h5py.File(train_hdf_, 'r')

        def generate_sample_clip(num_frames, seq_len):
            frame_indices = list(range(num_frames))
            if num_frames < seq_len:
                strip_frames = frame_indices + [frame_indices[-1]] * (seq_len - num_frames)
                sample_clip = [strip_frames[s * 1:(s + 1) * 1] for s in range(seq_len)]
            else:
                inter_val = math.ceil(num_frames / seq_len)
                strip_frames = frame_indices + [frame_indices[-1]] * (inter_val * seq_len - num_frames)
                sample_clip = [strip_frames[inter_val * s:inter_val * (s + 1)] for s in range(seq_len)]
            return np.array(sample_clip)

        sample_clip_ir = generate_sample_clip(num_ir, S)
        sample_clip_rgb = generate_sample_clip(num_rgb, S)

        def process_images(sample_clip, img_paths, hdf5_file):
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            selected_indices = sample_clip[np.arange(len(sample_clip)), idx]

            imgs, imgs_p = [], []
            for idx in selected_indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)
                img_path = img_path[16:]
                img_hdf5_key = img_path.replace('.jpg', '').split('/')
                img_p = hdf5_file[img_hdf5_key[3]][img_hdf5_key[4]][img_hdf5_key[5]][img_hdf5_key[6]][()]
                imgs.append(Tensor(img, dtype=mindspore.float32))
                imgs_p.append(img_p)
            imgs = ops.concat(imgs, axis=0)
            imgs_p = Tensor(np.stack(imgs_p, axis=0), dtype=mindspore.float32)
            return imgs, imgs_p

        imgs_ir, imgs_ir_p = process_images(sample_clip_ir, img_ir_paths, hdf5_file)
        imgs_rgb, imgs_rgb_p = process_images(sample_clip_rgb, img_rgb_paths, hdf5_file)

        return imgs_ir, imgs_ir_p, pid_ir, camid_ir, imgs_rgb, imgs_rgb_p, pid_rgb, camid_rgb


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        hdf5_file = h5py.File(test_hdf_, 'r')

        def generate_sample_clip(num_frames, seq_len):
            frame_indices = list(range(num_frames))
            if num_frames < seq_len:
                strip_frames = frame_indices + [frame_indices[-1]] * (seq_len - num_frames)
                sample_clip = [strip_frames[s * 1:(s + 1) * 1] for s in range(seq_len)]
            else:
                inter_val = math.ceil(num_frames / seq_len)
                strip_frames = frame_indices + [frame_indices[-1]] * (inter_val * seq_len - num_frames)
                sample_clip = [strip_frames[inter_val * s:inter_val * (s + 1)] for s in range(seq_len)]
            return np.array(sample_clip)

        sample_clip_ir = generate_sample_clip(num, S)

        def process_images(sample_clip, img_paths, hdf5_file):
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            selected_indices = sample_clip[np.arange(len(sample_clip)), idx]

            imgs, imgs_p = [], []
            for idx in selected_indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)
                img_path = img_path[16:]
                img_hdf5_key = img_path.replace('.jpg', '').split('/')
                img_p = hdf5_file[img_hdf5_key[3]][img_hdf5_key[4]][img_hdf5_key[5]][img_hdf5_key[6]][()]
                imgs.append(Tensor(img, dtype=mindspore.float32))
                imgs_p.append(img_p)
            imgs = ops.concat(imgs, axis=0)
            imgs_p = Tensor(np.stack(imgs_p, axis=0), dtype=mindspore.float32)
            return imgs, imgs_p

        imgs_ir, imgs_ir_p = process_images(sample_clip_ir, img_paths, hdf5_file)

        return imgs_ir, imgs_ir_p, pid, camid
