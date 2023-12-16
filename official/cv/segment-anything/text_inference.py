import argparse
import os

import cv2
import numpy as np

import mindspore as ms
from mindformers import AutoProcessor
from omegaconf import OmegaConf

from segment_anything.build_sam import sam_model_registry, create_model
from segment_anything.dataset.transform import TransformPipeline, ImageNorm, ImageResizeAndPad
import matplotlib.pyplot as plt

from segment_anything.utils.utils import Timer
from segment_anything.utils.visualize import show_mask, show_box


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

        if args.text_model_type == 'blip2':
            text_model_type = 'blip2_stage1_classification'
        elif args.text_model_type == 'clip':
            text_model_type = 'clip_vit_l_14@336'
        else:
            raise NotImplementedError
        processor = AutoProcessor.from_pretrained(text_model_type)
        tokenizer = processor.tokenizer
        input_ids = tokenizer([args.text_prompt], max_length=77, padding="max_length", return_tensors="ms")["input_ids"].unsqueeze(0)  # b, n, 7

    # Step2: inference
    with Timer('model inference'):
        with Timer('load weight and build net'):
            if args.text_model_type == 'blip2':
                text_encoder = dict(type='blip2_stage1_classification')
            elif args.text_model_type == 'clip':
                text_encoder = dict(type="clip_vit_l_14@336", feature_dim=768)
            else:
                raise NotImplementedError
            # network = create_model(OmegaConf.create({"text_encoder": dict(type="clip_vit_l_14@336", feature_dim=768), "type": args.model_type, "checkpoint": args.checkpoint, "enable_text_encoder": True, }))
            network = create_model(OmegaConf.create({"text_encoder": text_encoder, "type": args.model_type, "checkpoint": args.checkpoint, "enable_text_encoder": True, }))
        ms.amp.auto_mixed_precision(network=network, amp_level=args.amp_level)
        print(f'prompt is: {args.text_prompt}')
        mask_logits = network(image, text_ids=input_ids)[0]   # (1, 1, 1024, 1024)

    # Step3: post-process
    with Timer('post-process'):
        mask_logits = mask_logits.asnumpy()[0, 0] > 0.0
        mask_logits = mask_logits.astype(np.uint8)
        final_mask = cv2.resize(mask_logits[:origin_hw[2], :origin_hw[3]], tuple((origin_hw[1], origin_hw[0])),
                                interpolation=cv2.INTER_CUBIC)

    # Step4: visualize
    plt.imshow(image_np)
    show_mask(final_mask, plt.gca())
    save_path = os.path.basename(args.image_path) + '_infer.jpg'
    plt.savefig(save_path)
    print(f'finish saving inference image at {save_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./images/truck.jpg', help="Path to an input image.")
    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of sam model to load, in ['vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--text-model-type",
        type=str,
        default='blip2',
        help="The type of text model to load, in ['clip', 'blip2']",
    )

    parser.add_argument(
        "--text-prompt",
        type=str,
        default='wheels',
        help="Text prompt",
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
