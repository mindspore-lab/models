import os
import argparse
import ast
import cv2
import mindspore as ms
import numpy as np
from mindspore import nn

from src import get_network

from src.dataset.transforms_factory import create_transform
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env
from src.utils.eval_utils import DetectionEngine


def get_args_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/faster_rcnn/faster_rcnn_resnet50_fpn_1x.yml"),
        help="Config file path",
    )
    parser.add_argument("--seed", type=int, default=1234, help="runtime seed")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ckpt_path", type=str, default="", help="pre trained weights path")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")
    parser.add_argument("--imgs", type=str, default="imgs",
                        help="images or images folder, the image must be gpj or png image.")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")
    args, _ = parser.parse_known_args()
    return args


def get_img_list(path):
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a exist path!")
    img_paths = []
    if os.path.isdir(path):
        for img_path in os.listdir(path):
            img_path = os.path.join(path, img_path)
            name, suffix = os.path.splitext(img_path)
            if os.path.isfile(img_path) and suffix in [".jpg", ".png"]:
                img_paths.append(img_path)
        if not img_paths:
            raise ValueError(f"{path} is not a valid image folder!")
    elif os.path.isfile(path) and (path.endswith(".jpg") or path.endswith(".png")):
        img_paths.append(path)
    else:
        raise ValueError(f"{path} must be images or images folder, the image must be gpj or png image.")
    return img_paths


def get_transforms(data_config):
    trans_config = getattr(data_config, "eval_transforms", data_config)
    item_transforms = getattr(trans_config, "item_transforms", [])
    transforms_name_list = []
    for transform in item_transforms:
        transforms_name_list.extend(transform.keys())
    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        transform = create_transform(item_transforms[i])
        transforms_list.append(transform)
    return transforms_list


def img_preprocess(img_path, img_transforms):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_bbox = np.zeros((1, 4), np.float32)
    gt_class = np.zeros((1, 1), np.int32)
    ori_shape = np.array([img.shape[:2]])
    for transforms in img_transforms:
        img, _, _ = transforms(img, gt_bbox, gt_class)
    return img[None, ...], ori_shape


def draw(img_path, predicts, draw_colors, names, save_dir="./output", seg=False):
    im = cv2.imread(img_path)
    im_name = os.path.join(save_dir, os.path.basename(img_path))
    for data in predicts:
        label = names[int(data["label_id"])]
        color = draw_colors[int(data["label_id"])]
        bbox = data["bbox"]
        score = data["score"]
        text = f"{label}: {score:.2f}"
        x_l, y_t, x_r, y_b = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(color), 2)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(color), -1)
        cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if seg:
            mask = data["segmentation"].astye(im.dtype)
            im = (0.7 * im + 0.3 * mask).astype(np.uint8)
    cv2.imwrite(im_name, im)


if __name__ == "__main__":
    args = get_args_infer()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)
    draw_colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                   for _ in range(config.data.nc)]
    network = get_network(config)
    ms.load_checkpoint(config.ckpt_path, network)
    detection_engine = DetectionEngine(config)
    if config.mix:
        network.to_float(ms.float32)
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.to_float(ms.float16)
    img_list = get_img_list(config.imgs)
    img_transforms = get_transforms(config.data)
    for img_path in img_list:
        img, ori_shape = img_preprocess(img_path, img_transforms)
        detection_engine.input_shape = img.shape[2:]
        prediction = network.predict(ms.Tensor(img))
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = prediction.asnumpy()
        predicts = detection_engine.detection(prediction, ori_shape, [0])
        print(img_path, predicts)
        draw(img_path, predicts, draw_colors, config.data.names, save_dir="./output", seg=False)
