import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2

import mindspore as ms

from segment_anything.utils.visualize import show_mask, show_points, show_box


def main(args: argparse.Namespace):
    ms.context.set_context(mode=args.mode, device_target=args.device)

    from segment_anything import sam_model_registry, SamPredictor
    model_type = args.model_type
    sam_checkpoint = args.checkpoint

    print(f'running with {model_type}')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    predictor = SamPredictor(sam)

    image = cv2.imread('images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_dir = 'work_dirs'
    os.makedirs(out_dir, exist_ok=True)

    s = time.time()
    predictor.set_image(image)
    e = time.time()
    print(f'image encode time: {e-s:.1f}s')

    # predict one object a time
    if args.prompt_type == 'point':
        predict_with_point(predictor, image, args)
    elif args.prompt_type == 'box':
        predict_with_box(predictor, image, args)
    elif args.prompt_type == 'point_box':
        predict_with_point_and_box(predictor, image, args)

    # predict multiple objects in a batch way
    elif args.prompt_type == 'batch_point':
        predict_with_batch_point(predictor, image, args)
    elif args.prompt_type == 'batch_box':
        predict_with_batch_box(predictor, image, args)
    elif args.prompt_type == 'batch_point_box':
        predict_with_batch_point_and_box(predictor, image, args)
    else:
        raise NotImplementedError


def predict_with_point(predictor, image, args: argparse.Namespace):
    # predict the first point
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    s1 = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    s2 = time.time()
    print(f'prompting with points takes: {s2-s1:.1f}s')
    print(f'out shape: mask {masks.shape}, logits {logits.shape}, scores {scores.shape}')
    os.makedirs(args.output_dir, exist_ok=True)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        path = os.path.join(args.output_dir, f'mask_{i+1}.jpg')
        plt.savefig(path)
        print(f'saving mask at {path}')
        if args.enable_plt_visual:
            plt.show()

    # predict the second and third points
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 0])

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    print(f'mask input shape {mask_input.shape}')
    s3 = time.time()
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    s4 = time.time()
    print(f'prompting with mask tasks: {s4-s3:.1f}s')

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    path = os.path.join(args.output_dir, f'two_point.jpg')
    print(f'saving mask at {path}')
    plt.savefig(path)
    if args.enable_plt_visual:
        plt.show()


def predict_with_box(predictor, image, args: argparse.Namespace):
    input_box = np.array([425, 600, 700, 875])
    s1 = time.time()
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    s2 = time.time()
    print(f'prompting with box takes: {s2-s1:.1f}s')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    path = os.path.join(args.output_dir, f'box.jpg')
    print(f'saving mask at {path}')
    plt.savefig(path)
    if args.enable_plt_visual:
        plt.show()


def predict_with_point_and_box(predictor, image, args: argparse.Namespace):
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    path = os.path.join(args.output_dir, f'point_and_box.jpg')
    print(f'saving mask at {path}')
    plt.savefig(path)
    if args.enable_plt_visual:
        plt.show()


def predict_with_batch_point(predictor, image, args: argparse.Namespace):
    # predict the first point
    input_point = np.array([[[500, 375]], [[1125, 625]]])
    input_label = np.array([[1], [1]])

    point_coords = predictor.transform.apply_coords(input_point, predictor.original_size)
    coords_ms = ms.Tensor(point_coords, dtype=ms.float32)
    labels_ms = ms.Tensor(input_label, dtype=ms.int32)

    s1 = time.time()
    masks, scores, logits = predictor.predict_mask_and_iou(
        point_coords=coords_ms,
        point_labels=labels_ms,
        multimask_output=False,
    )
    masks = masks.asnumpy()
    scores = scores.asnumpy()
    logits = logits.asnumpy()
    s2 = time.time()
    print(f'prompting with batch points takes: {s2-s1:.1f}s')
    print(f'out shape: mask {masks.shape}, logits {logits.shape}, scores {scores.shape}')
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask[0]
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    path = os.path.join(args.output_dir, f'batch_point.jpg')
    plt.savefig(path)
    print(f'saving mask at {path}')
    if args.enable_plt_visual:
        plt.show()


def predict_with_batch_box(predictor, image, args: argparse.Namespace):
    input_boxes = np.array([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ])
    box = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
    box_ms = ms.Tensor(box, dtype=ms.float32)

    s1 = time.time()
    masks, _, _ = predictor.predict_mask_and_iou(
        point_coords=None,
        point_labels=None,
        boxes=box_ms,
        multimask_output=False,
    )
    masks = masks.asnumpy()
    s2 = time.time()
    print(f'prompting with box takes: {s2-s1:.1f}s')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i, (mask, box) in enumerate(zip(masks, input_boxes)):
        mask = mask[0]
        show_mask(mask, plt.gca(), random_color=True)
        show_box(box, plt.gca())
    plt.axis('off')
    path = os.path.join(args.output_dir, f'batch_box.jpg')
    print(f'saving mask at {path}')
    plt.savefig(path)
    if args.enable_plt_visual:
        plt.show()


def predict_with_batch_point_and_box(predictor, image, args: argparse.Namespace):
    input_boxes = np.array([[425, 600, 700, 875],
                            [1360, 525, 1680, 780]])
    input_points = np.array([[[575, 750]],
                             [[1525, 670]]])
    input_labels = np.array([[0], [1]])
    box = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
    point_coords = predictor.transform.apply_coords(input_points, predictor.original_size)
    box_ms = ms.Tensor(box, dtype=ms.float32)
    coords_ms = ms.Tensor(point_coords, dtype=ms.float32)
    labels_ms = ms.Tensor(input_labels, dtype=ms.int32)

    masks, _, _ = predictor.predict_mask_and_iou(
        point_coords=coords_ms,
        point_labels=labels_ms,
        boxes=box_ms,
        multimask_output=False,
    )
    masks = masks.asnumpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_points, input_labels, plt.gca())

    for i, (mask, box) in enumerate(zip(masks, input_boxes)):
        mask = mask[0]
        show_mask(mask, plt.gca(), random_color=True)
        show_box(box, plt.gca())

    plt.axis('off')
    path = os.path.join(args.output_dir, f'point_and_box.jpg')
    print(f'saving mask at {path}')
    plt.savefig(path)
    if args.enable_plt_visual:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Runs mask generation with prompts on an input image"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='./prompt_test_img',
        help=(
            "Path to the directory where the picture with masks will be output."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    # models/sam_vit_h-c72f8ba1.ckpt
    # models/sam_vit_l-1b460f38.ckpt
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='./models/sam_vit_b-35e4849c.ckpt',
        help="The checkpoint path to model",
    )

    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")

    parser.add_argument("--mode", type=int, default=0, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    parser.add_argument(
        "--enable-plt-visual",
        action="store_true",
        help=(
            "Button to enable matplot visualization."
        ),
    )

    parser.add_argument(
        "--prompt-type",
        type=str,
        default='point',
        help="The type of prompt to load, in ['point', 'box', 'point_box']",
    )
    args = parser.parse_args()
    print(args)
    main(args)