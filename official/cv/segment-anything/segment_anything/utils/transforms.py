import numpy as np
import mindspore as ms
from mindspore import ops, mint
from PIL import Image

from copy import deepcopy
from typing import Tuple


def ndarray_to_pil(npimg):
    """
    numpy ndarray to pillow image
    """
    # suppose numpy image in H*W*3 or H*W format
    assert npimg.dtype == np.uint8
    mode = 'RGB'
    if len(npimg.shape) == 3 and npimg.shape[2] == 3:
        mode = "RGB"
    if len(npimg.shape) == 2:
        mode = 'L'

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)


def resize_no_alias(image, new_hw):
    """
    resize image with no alias

    apply resize with pillow to remove aliasing. See below link for more info about aliasing
    https://stackoverflow.com/questions/60949936/why-bilinear-scaling-of-images-with-pil-and-pytorch-produces-different-results

    Args:
        image (np.ndarray): in shape [h, w, c] or [h, w].
        new_hw (tuple): new shape to resize to, in [h, w] format.
    """

    pil_img = ndarray_to_pil(image)
    pil_img = pil_img.resize(tuple(new_hw[::-1]), resample=Image.Resampling.BILINEAR)
    np_img = np.array(pil_img)

    return np_img


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # TODO move image norm and pad inside this functions; return img and size before padding
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)

        # apply resize with pillow to remove aliasing. See below link for more info about aliasing
        # https://stackoverflow.com/questions/60949936/why-bilinear-scaling-of-images-with-pil-and-pytorch-produces-different-results
        pil_img = ndarray_to_pil(image)
        pil_img = pil_img.resize(tuple(target_size[::-1]), resample=Image.Resampling.BILINEAR)
        np_img = np.array(pil_img)

        return np_img

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(np.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_ms(self, image: ms.Tensor) -> ms.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        # TODO note original version has antialias=True, ref:
        # https://stackoverflow.com/questions/60949936/why-bilinear-scaling-of-images-with-pil-and-pytorch-produces-different-results
        return mint.interpolate(
            image, target_size, mode="bilinear", align_corners=False
        )

    def apply_coords_ms(
        self, coords: ms.Tensor, original_size: Tuple[int, ...]
    ) -> ms.Tensor:
        """
        Expects a ms tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(ms.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_ms(
        self, boxes: ms.Tensor, original_size: Tuple[int, ...]
    ) -> ms.Tensor:
        """
        Expects a ms tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_ms(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
