import argparse

import cv2
import numpy as np

import mindspore as ms
from mindspore import nn

from segment_anything.dataset.transform import TransformPipeline, ImageNorm, ImageResizeAndPad
import matplotlib.pyplot as plt

from segment_anything.utils.utils import Timer
from segment_anything.utils.visualize import show_box, show_mask


def infer(args):
    ms.context.set_context(mode=args.mode, device_target=args.device)

    # Step1: data preparation
    with Timer('preprocess'):
        transform_list = [
            ImageResizeAndPad(target_size=1024, apply_mask=False),
            ImageNorm(),
        ]
        transform_pipeline = TransformPipeline(transform_list)

        image_path = args.image_path
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        boxes_np = np.array([[425, 600, 700, 875]])

        transformed = transform_pipeline(dict(image=image_np, boxes=boxes_np))
        image, boxes, origin_hw = transformed['image'], transformed['boxes'], transformed['origin_hw']
        # batch_size for speed test
        # image = ms.Tensor(np.expand_dims(image, 0).repeat(8, axis=0))  # b, 3, 1023
        # boxes = ms.Tensor(np.expand_dims(boxes, 0).repeat(8, axis=0))  # b, n, 4
        image = ms.Tensor(image).unsqueeze(0)  # b, 3, 1023
        boxes = ms.Tensor(boxes).unsqueeze(0)  # b, n, 4

    # Step2: inference
    with Timer('model inference'):
        with Timer('load weight and build net'):
            graph = ms.load(args.model_path)
            network = nn.GraphCell(graph)
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
    plt.savefig(args.image_path + '_mindir_infer.jpg')
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
        "--model-path",
        type=str,
        default='./models/sam_vit_b.mindir',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")
    parser.add_argument("--amp_level", type=str, default="O2", help="auto mixed precision level O0, O2.")
    parser.add_argument("--mode", type=int, default=0, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    args = parser.parse_args()
    print(args)
    infer(args)
