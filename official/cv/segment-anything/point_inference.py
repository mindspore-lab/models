import argparse
import os

import cv2
import numpy as np

import mindspore as ms

from segment_anything.build_sam import sam_model_registry
from segment_anything.dataset.transform import TransformPipeline, ImageNorm, ImageResizeAndPad, \
    BinaryMaskFromInstanceSeg, BoxFormMask
import matplotlib.pyplot as plt

from segment_anything.utils.utils import Timer
from segment_anything.utils.visualize import show_mask, show_box, show_points


def infer(args):
    ms.context.set_context(mode=args.mode, device_target=args.device)

    # Step1: data preparation
    with Timer('preprocess'):
        image_path = args.image_path
        image_np = np.load(image_path).astype(np.float32)  # (1024, 1024, 3)

        # points_np = np.array([[[564, 498]]])  # box center
        # points_np = np.array([[[550, 505]]])
        # points_np = np.array([[[575, 475]]])
        points_np = np.array([[[575, 550]]])
        labels_np = np.array([[1]])

        # batch_size for speed test
        # image = ms.Tensor(np.expand_dims(image, 0).repeat(8, axis=0))  # b, 3, 1023
        # boxes = ms.Tensor(np.expand_dims(boxes, 0).repeat(8, axis=0))  # b, n, 4
        image = ms.Tensor(image_np.transpose([2, 0, 1])).unsqueeze(0)  # b, 3, 1024, 1024
        points = ms.Tensor(points_np).unsqueeze(0)  # b, bp, n, 2
        labels = ms.Tensor(labels_np).unsqueeze(0)  # b, bp, n
        point_and_label = ms.mutable((points, labels))

    # Step2: inference
    with Timer('model inference'):
        with Timer('load weight and build net'):
            network = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        ms.amp.auto_mixed_precision(network=network, amp_level=args.amp_level)
        mask_logits = network(image, points=point_and_label, multimask_output=True, output_best_mask=True)[0]   # (1, 1, 1024, 1024)

    # Step3: post-process
    with Timer('post-process'):
        mask_logits = mask_logits.asnumpy()[0, 0] > 0.0
        mask_logits = mask_logits.astype(np.uint8)
        final_mask = mask_logits

    # Step4: visualize
    plt.imshow(image_np)
    show_points(points_np[0], labels_np[0], plt.gca(), marker_size=100)
    show_mask(final_mask, plt.gca())
    plt.savefig(args.image_path + '_point_infer.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./datasets/FLARE22Train_processed/train/images/CT_Abd_FLARE22_Tr_0001-000.npy', help="Path to an input image.")
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
    parser.add_argument("--amp_level", type=str, default="O0", help="auto mixed precision level O0, O2.")
    parser.add_argument("--mode", type=int, default=0, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    args = parser.parse_args()
    print(args)
    infer(args)
