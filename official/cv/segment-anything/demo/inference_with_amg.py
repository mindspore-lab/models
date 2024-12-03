import argparse
import os
import time

import matplotlib.pyplot as plt
import cv2

import mindspore as ms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.visualize import show_anns


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace):
    ms.context.set_context(mode=args.mode, device_target=args.device, pynative_synchronize=False)

    checkpoints = dict(vit_b="models/sam_vit_b-35e4849c.ckpt",
                       vit_l="models/sam_vit_l-1b460f38.ckpt",
                       vit_h="models/sam_vit_h-c72f8ba1.ckpt")
    model_type = args.model_type
    sam_checkpoint = checkpoints[model_type]
    print(f'running with {model_type}')
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    # generate more mask
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )

    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, [image.shape[1]//5, image.shape[0]//5])
    s1 = time.time()
    masks = mask_generator.generate(image)
    s2 = time.time()
    print(f'amg time: {s2-s1:.1f}s')
    print('number of mask: ', len(masks))
    print('mask keys: ', masks[0].keys())

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, pad_inches=0)
    print(f'saving fig at {args.output}')

    if args.enable_plt_visual:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default='https://github.com/user-attachments/assets/3280f801-f952-4761-b761-d02c71058086',
        help="Path to a single input image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default='./amg_test.jpg',
        help=(
            "Path where the picture with masks will be output."
        ),
    )

    parser.add_argument(
        "--enable-plt-visual",
        action="store_true",
        help=(
            "Button to enable matplot visualization."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")

    parser.add_argument("--mode", type=int, default=0, help="MindSpore context mode. 0 for graph, 1 for pynative.")

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )

    args = parser.parse_args()
    print(args)
    main(args)
