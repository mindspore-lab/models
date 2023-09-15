import argparse
import os

import cv2
import numpy as np

import mindspore as ms

from segment_anything.build_sam import sam_model_registry
from segment_anything.dataset.transform import TransformPipeline, ImageNorm
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import time

from use_sam_with_promts import show_mask, show_box


class Timer:
    def __init__(self, name=''):
        self.name = name
        self.start = 0.0
        self.end = 0.0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f'{self.name} cost time {self.end - self.start:.3f}')


class ImageResizeAndPad:

    def __init__(self, target_size):
        """
        Args:
            target_size (int): target size of model input (1024 in sam)
        """
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __call__(self, result_dict):
        """
        Resize input to the long size and then pad it to the model input size (1024*1024 in sam).
        Pad masks and boxes to a fixed length for graph mode
        Required keys: image, masks, boxes
        Update keys: image, masks, boxes
        Add keys:
            origin_hw (np.array): array with shape (4), represents original image height, width
            and resized height, width, respectively. This array record the trace of image shape transformation
            and is used for visualization.
            image_pad_area (Tuple): image padding area in h and w direction, in the format of
            ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        """

        image = result_dict['image']
        boxes = result_dict['boxes']

        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        resized_h, resized_w, _ = image.shape

        # Pad image and masks to the model input
        h, w, c = image.shape
        max_dim = max(h, w)  # long side length
        assert max_dim == self.target_size
        # pad 0 to the right and bottom side
        pad_h = max_dim - h
        pad_w = max_dim - w
        img_padding = ((0, pad_h), (0, pad_w), (0, 0))
        image = np.pad(image, pad_width=img_padding, constant_values=0)  # (h, w, c)

        # Adjust bounding boxes
        boxes = self.transform.apply_boxes(boxes, (og_h, og_w)).astype(np.float32)

        result_dict['origin_hw'] = np.array([og_h, og_w, resized_h, resized_w], np.int32)  # record image shape trace for visualization
        result_dict['image'] = image
        result_dict['boxes'] = boxes
        result_dict['image_pad_area'] = img_padding[:2]

        return result_dict


def infer(args):
    ms.context.set_context(mode=args.mode, device_target=args.device)

    # Step1: data preparation
    with Timer('preprocess'):
        transform_list = [
            ImageResizeAndPad(target_size=1024),
            ImageNorm(),
        ]
        transform_pipeline = TransformPipeline(transform_list)

        image_path = args.image_path
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        boxes_np = np.array([[425, 600, 700, 875]])

        transformed = transform_pipeline(dict(image=image_np, boxes=boxes_np))
        image, boxes, origin_hw = transformed['image'], transformed['boxes'], transformed['origin_hw']
        image = ms.Tensor(image).unsqueeze(0)  # b, 3, 1023
        boxes = ms.Tensor(boxes).unsqueeze(0)  # b, n, 4

    # Step2: inference
    with Timer('model inference'):
        with Timer('load weight and build net'):
            network = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        ms.amp.auto_mixed_precision(network=network, amp_level=args.amp_level)
        mask_logits = network(image, boxes)[0]   # (1, 1, 1024, 1024)

    with Timer('Second time inference'):
        mask_logits = network(image, boxes)[0]  # (1, 1, 1024, 1024)

    # Step3: post-process
    with Timer('post-process'):
        mask_logits = mask_logits.asnumpy()[0, 0] > 0.0
        mask_logits = mask_logits.astype(np.uint8)
        final_mask = cv2.resize(mask_logits[:origin_hw[2], :origin_hw[3]], tuple((origin_hw[1], origin_hw[0])),
                                interpolation=cv2.INTER_CUBIC)

    # Step4: visualize
    plt.imshow(image_np)
    show_box(boxes_np[0], plt.gca())
    show_mask(final_mask, plt.gca())
    plt.savefig(args.image_path + '_infer.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./images/truck.jpg', help="Path to an input image.")
    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default='./models/sam_vit_b-35e4849c.ckpt',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")
    parser.add_argument("--amp_level", type=str, default="O2", help="auto mixed precision level O0, O2.")
    parser.add_argument("--mode", type=int, default=0, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    args = parser.parse_args()
    print(args)
    infer(args)
