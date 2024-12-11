import random
import math
import numpy as np
from mindspore import Tensor
import mindspore.common.dtype as mstype

class ChannelAdap:
    """
    Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the transformation will be performed.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (C, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """
        idx = random.randint(0, 3)
        if idx == 0:
            # Randomly select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # Randomly select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # Randomly select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        return img

class ChannelAdapGray:
    """
    Adaptive selects a channel or two channels, or convert to grayscale based on probability.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (C, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """
        idx = random.randint(0, 3)
        if idx == 0:
            # Randomly select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # Randomly select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # Randomly select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            # Convert to grayscale with a probability
            if random.uniform(0, 1) <= self.probability:
                gray = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = gray
                img[1, :, :] = gray
                img[2, :, :] = gray
        return img

class ChannelRandomErasing:
    """
    Randomly selects a rectangle region in an image and erases its pixels.
    Args:
        probability: The probability that the transformation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (C, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """
        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.shape[1] * img.shape[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img
