import errno
import os
from collections import defaultdict
import numbers
import mindspore
import numpy as np
import sys
import os.path as osp
import scipy.io as scio
import cv2
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore import Tensor, ops
from mindspore.common import Parameter

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = f.read().splitlines()
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for label in unique_label_color:
        tmp_pos = [idx for idx, val in enumerate(train_color_label) if val == label]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for label in unique_label_thermal:
        tmp_pos = [idx for idx, val in enumerate(train_thermal_label) if val == label]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


def GenCamIdx(gall_img, gall_label, mode):
    camIdx = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]
    gall_cam = [int(img[-10]) for img in gall_img]

    sample_pos = []
    unique_label = np.unique(gall_label)
    for label in unique_label:
        for cam in camIdx:
            id_pos = [idx for idx, val in enumerate(gall_label) if val == label and gall_cam[idx] == cam]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    return np.array([int(img[-10]) for img in gall_img])


class IdentitySampler(Cell):
    """Sample person identities evenly in each batch.
    Args:
        train_color_label, train_thermal_label: labels of two modalities
        color_pos, thermal_pos: positions of each identity
        batchSize: the number of pids
    """
    def __init__(self, train_thermal_label, train_color_label, color_pos, thermal_pos, num_pos, batchSize):
        super(IdentitySampler, self).__init__()
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.minimum(len(train_color_label), len(train_thermal_label))
        indices1, indices2 = [], []

        for _ in range(int(N / (batchSize * num_pos))):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for idx in batch_idx:
                sample_color = np.random.choice(color_pos[idx], num_pos)
                sample_thermal = np.random.choice(thermal_pos[idx], num_pos)
                indices1.extend(sample_color)
                indices2.extend(sample_thermal)

        self.index1 = np.array(indices1)
        self.index2 = np.array(indices2)
        self.N = len(self.index1)

    def construct(self):
        return Tensor(np.arange(len(self.index1)), dtype=mindspore.int32)

    def __len__(self):
        return self.N


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger:
    """Write console output to external text file."""
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    if cuda:
        # MindSpore seeds can be set via `context` configurations
        print(f"MindSpore doesn't require explicit cuda seed configuration. Setting numpy seed: {seed}")


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size[1], output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = [0, (dst_w - 1) * -0.5]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5] + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        target_logits.append(target_logit)
    return np.stack(target_logits, axis=2)
