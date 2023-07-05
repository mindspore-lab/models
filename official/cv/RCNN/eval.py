import os
import argparse
import ast
import traceback
import mindspore as ms
from mindspore import nn

from src import get_network
from src.utils import logger
from src.dataset.loader import create_dataloader
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env, clear
from src.utils.eval_utils import run_eval


def get_args_eval(parents=None):
    parser = argparse.ArgumentParser(description="Eval", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml"),
        help="Config file path",
    )
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--num_parallel_workers", type=int, default=4, help="num parallel worker for dataloader")
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")
    parser.add_argument("--ckpt_path", type=str, default="", help="pre trained weights path")
    parser.add_argument("--batch_size", type=int, default=1, help="total batch size for all device")
    parser.add_argument("--eval_parallel", type=ast.literal_eval, default=True, help="run eval")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")

    # profiling
    parser.add_argument("--run_profilor", type=ast.literal_eval, default=False, help="run profilor")

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--data_dir", type=str, default="/cache/data", help="ModelArts: obs path to dataset folder")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args_eval()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)
    logger.info(config)
    config.backbone.pretrained = ""
    network = get_network(config)
    if config.mix:
        network.to_float(ms.float32)
        network.backbone.to_float(ms.float16)
        network.rpn_head.rpn_feat.to_float(ms.float16)
        network.bbox_head.head.to_float(ms.float16)
        network.bbox_head.roi_extractor.to_float(ms.float16)
        if config.net == "MaskRCNN":
            network.mask_head.head.to_float(ms.float16)
            network.mask_head.roi_extractor.to_float(ms.float16)
            network.mask_head.mask_fcn_logits.to_float(ms.float16)
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.Dense)):
                cell.to_float(ms.float16)

    eval_dataloader, coco_dataset = create_dataloader(
        data_config=config.data,
        task="eval",
        per_batch_size=config.batch_size,
        rank=config.rank,
        rank_size=config.rank_size,
        shuffle=False,
        drop_remainder=True,
    )

    if len(coco_dataset) % (config.rank_size * config.batch_size) != 0 and config.infer.eval_parallel:
        logger.warning(
            f"dataset len {len(coco_dataset)} cannot split to {config.rank_size} "
            f"with batch_size {config.batch_size}, it will drop some samples."
        )
    try:
        ckpt_list = []
        if os.path.isdir(config.ckpt_path):
            ckpt_list = []
            for f in os.listdir(config.ckpt_path):
                if f.endswith(".ckpt"):
                    ckpt_list.append(os.path.join(config.ckpt_path, f))
        elif os.path.isfile(config.ckpt_path) and config.ckpt_path.endswith(".ckpt"):
            ckpt_list = [config.ckpt_path]
        else:
            raise ValueError("{config.ckpt_path} is not a ckpt file or dir")
        for ckpt in ckpt_list:
            logger.info(f"eval {ckpt}")
            ms.load_checkpoint(ckpt, network)
            run_eval(config, network, eval_dataloader)
    except:
        traceback.print_exc()
    finally:
        clear(config)
