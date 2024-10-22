
import mindspore.nn as nn
import mindspore
import pdb

import numpy as np, cv2 as cv, scipy
from scipy import signal
import collections

import imageio


# Preprocessing functions
def preprocess(image):
    # Converts image range from [0,1] to [-1,1]
    return image * 2 - 1


def deprocess(image):
    # [-1,1] --> [0,1]
    return (image + 1) / 2


def preprocessLr(image):
    identity = nn.Identity()
    return identity(image)


def deprocessLr(image):
    identity = nn.Identity()
    return identity(image)








# Different processing functions

def pixelshuffle(inputs, scale=2):
    shuffle = nn.PixelShuffel(2)
    return shuffle


def upscale_four(inputs):
    upsample = nn.Upsample(scale_factor=4, mode="bilinear")
    return upsample(inputs)


def bicubic_four(inputs):
    upsample = nn.Upsample(scale_factor=4, mode="bicubic")
    return upsample(inputs)









def compute_psnr(ref, target):
    ref = ref.float()
    target = target.float()
    diff = target - ref
    sqr = mindspore.Tensor.multiply(diff, diff)
    err = sqr.sum()
    v = diff.shape[0] * diff.shape[1] * diff.shape[2] * diff.shape[3]
    mse = err / v
    psnr = 10. * (mindspore.ops.log(255. * 255. / mse) / mindspore.ops.log(mindspore.tensor(10.)))
    return psnr


# Defining the VGG model for layer loss



# Upsample functions

def gaussian_2dkernel(size=5, sig=1.):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return (gkern2d / gkern2d.sum())


# # Loading checkpoint
# def load_ckpt(checkpoint, model):
#     return model.load_state_dict(mindspore.load(checkpoint))


# Functions for saving images and gifs

def save_as_gif(tensor, filepath):
    img = tensor.float().numpy() * 255.
    images = np.transpose(img.astype(np.uint8), (0, 2, 3, 1))
    imageio.mimsave(filepath, images)


def save_img(out_path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:, :, ::-1])

def _rgb2ycbcr(img, maxVal=255):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr

def to_uint8(x, vmin, vmax):
##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    x = x.astype('float32')
    x = (x-vmin)/(vmax-vmin)*255 # 0~255
    return np.clip(np.round(x), 0, 255)


def cut_image(image, vmin, vmax):
    image = np.maximum(image, vmin)
    image = np.minimum(image, vmax)
    return image

def psnr(img_true, img_pred):
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
    diff = Y_true - Y_pred
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 10 * np.log10(255. / rmse)

from skimage.measure import compare_ssim
def ssim(img_true, img_pred):  ##### SSIM #####
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
    return compare_ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())


