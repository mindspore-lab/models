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
        # image_np = np.load('./datasets/FLARE22Train_processed/train/images/CT_Abd_FLARE22_Tr_0001-010.npy')
        # cv2.imwrite('./images/tumor2.png', image_np*255)
        transform_list = [
            ImageResizeAndPad(target_size=1024, apply_mask=False, apply_point=True, apply_box=False),
            ImageNorm(),
        ]
        transform_pipeline = TransformPipeline(transform_list)

        image_path = args.image_path
        image_np = cv2.imread(image_path) # (1024, 1024, 3)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # image = np.load(image_path)
        # cv2.imwrite('./images/tumor1.png', image_np*255)
        # image_npnew = cv2.imread('./images/tumor1.png')/255.0  # (1024, 1024, 3)
        # diff = image_npnew - image_np

        points_np = np.array([[[868, 622]], [[724, 466]], [[501, 543]], [[565, 505]], [[262, 456]]]) #  (bp, n, 2), in [w, h]
        labels_np = np.array([[1],[1],[1],[1],[1]])  # 1 for positive, 0 for negative  # bp, n
        # show_points(points_np.reshape(-1, 2), labels_np.reshape(-1), plt.gca(), marker_size=100)
        # batch_size for speed test
        transformed = transform_pipeline(dict(image=image_np, points=points_np))
        image, points, origin_hw = transformed['image'], transformed['points'], transformed['origin_hw']

        image = ms.Tensor(image).unsqueeze(0)  # b, 3, 1024, 1024
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
        mask_logits = mask_logits.asnumpy()[0] > 0.0
        mask_logits = mask_logits.astype(np.uint8)
        final_masks = []
        for m in mask_logits:
            final_mask = cv2.resize(m[:origin_hw[2], :origin_hw[3]], tuple((origin_hw[1], origin_hw[0])),
                                    interpolation=cv2.INTER_CUBIC)
            final_masks.append(final_mask)

    # Step4: visualize
    plt.imshow(image_np)
    show_points(points_np.reshape(-1, 2), labels_np.reshape(-1), plt.gca(), marker_size=100)
    for m in final_masks:
        show_mask(m, plt.gca())
    save_path = os.path.splitext(os.path.basename(args.image_path))[0] + '_infer.jpg'
    plt.savefig(save_path)
    print(f'finish saving inference image at {save_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./images/tumor2.png', help="Path to an input image.")
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
