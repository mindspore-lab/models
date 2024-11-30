"""
Arguments for inference.

Argument names are adopted from ppocr for easy usage transfer.
"""
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

    parser.add_argument("--mode", type=int, default=0, help="0 for graph mode, 1 for pynative mode ")
    parser.add_argument("--image_dir", type=str, required=True, help="image path or image directory")

    parser.add_argument(
        "--rec_algorithm",
        type=str,
        default="CRNN",
        choices=["CRNN", "RARE", "CRNN_CH", "RARE_CH", "SVTR", "SVTR_PPOCRv3_CH", "CAN"],
        help="recognition algorithm",
    )
    parser.add_argument(
        "--rec_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--rec_model_dir",
        type=str,
        help="directory containing the recognition model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )

    parser.add_argument(
        "--rec_image_shape",
        type=str,
        default="3, 32, 320",
        help="C, H, W for target image shape. max_wh_ratio=W/H will be used to control the maximum width after "
        '"aspect-ratio-kept" resizing. Set W larger for longer text.',
    )

    parser.add_argument(
        "--rec_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )
    parser.add_argument("--rec_batch_num", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=None,
        help="path to character dictionary. If None, will pick according to rec_algorithm and red_model_dir.",
    )

    parser.add_argument("--vis_font_path", type=str, default="docs/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

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

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
