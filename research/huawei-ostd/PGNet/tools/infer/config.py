import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_parser():
    parser = argparse.ArgumentParser(description="Inference Config Args")
    # params for prediction engine
    parser.add_argument("--mode", type=int, default=0, help="0 for graph mode, 1 for pynative mode ")
    parser.add_argument("--e2e_model_config", type=str, help="path to rec model yaml config")
    parser.add_argument("--image_dir", type=str, required=True, help="image path or image directory")

    # ======== End2End ========
    parser.add_argument(
        "--e2e_algorithm",
        type=str,
        default="PG",
        choices=["PG"],
        help="end2end detection & recognition algorithm.",
    )
    parser.add_argument(
        "--e2e_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--e2e_model_dir",
        type=str,
        default=None,
        help="directory containing the model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--e2e_limit_side_len", type=int, default=768, help="side length limitation for image resizing"
    )
    parser.add_argument(
        "--e2e_limit_type",
        type=str,
        default="max",
        choices=["min", "max"],
        help="limitation type for image resize. If min, images will be resized by limiting the minimum side length "
        "to `limit_side_len` (prior to accuracy). If max, images will be resized by limiting the maximum side "
        "length to `limit_side_len` (prior to speed). Default: max",
    )

    parser.add_argument(
        "--draw_img_save_dir",
        type=str,
        default="./inference_results",
        help="Dir to save visualization and detection/recogintion/system prediction results",
    )
    parser.add_argument(
        "--save_crop_res",
        type=str2bool,
        default=False,
        help="Whether to save images cropped from text detection results.",
    )
    parser.add_argument(
        "--crop_res_save_dir", type=str, default="./output", help="Dir to save the cropped images for text boxes"
    )
    parser.add_argument(
        "--visualize_output",
        type=str2bool,
        default=False,
        help="Whether to visualize results and save the visualized image.",
    )

    parser.add_argument("--warmup", type=str2bool, default=False)
    parser.add_argument("--ocr_result_dir", type=str, default=None, help="path or directory of ocr results")
    parser.add_argument(
        "--ser_algorithm",
        type=str,
        default="VI_LAYOUTXLM",
        choices=["VI_LAYOUTXLM", "LAYOUTXLM"],
        help="ser algorithm",
    )
    parser.add_argument(
        "--ser_model_dir",
        type=str,
        help="directory containing the ser model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--kie_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )
    parser.add_argument("--kie_batch_num", type=int, default=8)

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
