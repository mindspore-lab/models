import os
import argparse
import ast
import traceback
import numpy as np
import mindspore as ms

from src.modules.ocrnet import OCRNet
from src.modules.base_modules import MultiScaleInfer
from src.data.dataset_factory import create_dataset
from src.utils import logger
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env, clear
from src.utils.metrics import GetConfusionMatrix
from src.data.visualize import visualize


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/ocrnet/config_ocrnet_hrw48_16k.yml"),
        help="Config file path",
    )
    parser.add_argument("--seed", type=int, default=1234, help="runtime seed")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ckpt_path", type=str, default="", help="pre trained weights path")
    parser.add_argument("--visualize", type=ast.literal_eval, default=False, help="visualize when eval")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")
    parser.add_argument("--amp_level", type=str, default="O3", help="Supports ['O0', 'O1', 'O2', 'O3']. Default: 'O3'")

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


def run_eval(cfg, net, eval_dataset):
    net.set_train(False)
    num_classes = cfg.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes))
    item_count = 0
    data_loader = eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    get_confusion_matrix = GetConfusionMatrix(num_classes, cfg.data.ignore_label, cfg.rank_size)
    if cfg.visualize:
        import shutil
        from src.data.cityscapes import Cityscapes
        save_path = os.path.join(config.save_dir, f"images_{cfg.rank}")
        if os.path.exists(save_path):

            shutil.rmtree(save_path)
        os.makedirs(save_path)
        classes = Cityscapes().classes
        palette = Cityscapes().classes
    for i, data in enumerate(data_loader):
        img = data["image"]
        label = data["label"]
        pred = net(ms.Tensor(img)).asnumpy()
        pred = np.squeeze(pred).astype(np.uint8)
        label = np.squeeze(label.astype(np.uint8))
        if cfg.visualize:
            visualize(i, save_path, label, pred, classes, palette, ignore_label=config.data.ignore_label,
                      img_shape=None, keep_ratio=True, label_map=None)
        confusion_matrix += get_confusion_matrix(label, pred)
        item_count += cfg.rank_size
    logger.info(f"Total number of images: {item_count}")

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = tp / np.maximum(1.0, pos + res - tp)
    mean_iou = iou_array.mean()

    # Show results
    logger.info(f"=========== Evaluation Result ===========")
    logger.info(f"iou array: \n {iou_array}")
    logger.info(f"miou: {mean_iou}")


if __name__ == "__main__":
    args = get_args_train()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)
    logger.info(f"{config}")

    # Dataset
    dataset, num = create_dataset(
        config.data,
        batch_size=1,
        num_parallel_workers=config.data.num_parallel_workers,
        task="eval",
        group_size=config.rank_size,
        rank=config.rank,
    )

    # Network
    network = OCRNet(config).set_train(False)
    if config.mix:
        ms.amp.auto_mixed_precision(network, config.amp_level)
    ms.load_checkpoint(config.ckpt_path, network)
    eval_net = MultiScaleInfer(
        network,
        num_classes=config.num_classes,
        img_ratios=config.data.eval_transforms.img_ratios,
        flip=config.data.eval_transforms.flip,
        multi_out=len(config.loss_weight) > 1,
    )

    logger.info(f"success to load pretrained ckpt {config.ckpt_path}")
    try:
        run_eval(config, eval_net, dataset)
    except:
        traceback.print_exc()
    finally:
        clear(enable_modelarts=config.enable_modelarts, save_dir=config.save_dir, train_url=config.train_url)
