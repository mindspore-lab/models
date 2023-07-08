import os
import sys
import time
import json
import copy
from datetime import datetime
import cv2
import numpy as np
from typing import List, Union
import mindspore as ms
from . import logger

try:
    from third_party.fast_coco_eval_api import Fast_COCOeval as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from .nms import multiclass_nms


def run_eval(cfg, network, dataset, cur_epoch=0, cur_step=0):
    network.set_train(False)
    # init detection engine
    detection = DetectionEngine(cfg)
    save_prefix = None
    if cfg.infer.eval_parallel:
        save_prefix = os.path.join(cfg.save_dir, "eval_parallel")
    eval_wrapper = EvalWrapper(cfg, dataset, network, detection, save_prefix)
    logger.info("Start inference...")
    eval_wrapper.inference(cur_epoch=cur_epoch, cur_step=cur_step)
    logger.info("Finish inference...")
    network.set_train(True)


class EvalWrapper:
    def __init__(self, cfg, dataset, network, detection_engine, save_prefix):
        self.dataset = dataset
        self.network = network
        self.detection_engine = detection_engine
        self.eval_parallel: bool = cfg.infer.eval_parallel
        self.rank_id: int = cfg.rank
        self.data_list = []
        self.img_ids = []
        self.rank_size: int = cfg.rank_size
        self.save_prefix = os.path.join(cfg.save_dir, "eval") if save_prefix is None else save_prefix
        os.makedirs(self.save_prefix, exist_ok=True)
        self.file_path = ""
        self.dir_path = ""
        self.is_segment = cfg.data.is_segment

    def synchronize(self):
        load_lock = os.path.join(self.save_prefix, f"synchronize_{self.rank_id}")
        os.mknod(load_lock)
        # Each server contains 8 devices as most.
        while True:
            synchronize_f = True
            for i in range(self.rank_size):
                if not os.path.exists(os.path.join(self.save_prefix, f"synchronize_{i}")):
                    synchronize_f = False
            time.sleep(1)
            if synchronize_f:
                time.sleep(5)
                os.remove(load_lock)
                break

    def inference_step(self, idx, data):
        image = data["image"]
        img_shape = data["ori_shape"]
        img_id = data["im_id"]
        self.detection_engine.input_shape = image.shape[2:]
        prediction = self.network.predict(ms.Tensor(image))
        if isinstance(prediction, (tuple, list)):
            if self.is_segment:
                mask = prediction[1].asnumpy()
            prediction = prediction[0]
        prediction = prediction.asnumpy()
        img_shape = img_shape
        if self.eval_parallel:
            prediction_p, img_shape_p, img_id_p, mask_p = [], [], [], []
            for i in range(prediction.shape[0]):
                if img_id[i] in self.img_ids:
                    continue
                self.img_ids.append(img_id[i])
                prediction_p.append(prediction[i])
                img_shape_p.append(img_shape[i])
                img_id_p.append(img_id[i])
                if self.is_segment:
                    mask_p.append(mask[i])
            prediction = np.stack(prediction_p, 0)
            img_shape = np.stack(img_shape_p, 0)
            img_id = np.stack(img_id_p, 0)
            if self.is_segment:
                mask = np.stack(mask_p, 0)

        if prediction.shape[0] > 0:
            if not self.is_segment:
                mask = None
            data = self.detection_engine.detection(prediction, img_shape, img_id, mask)
            self.data_list.extend(data)

    def inference(self, cur_epoch=0, cur_step=0):
        self.network.set_train(False)
        dataset_size = self.dataset.get_dataset_size()
        logger.info("Start inference...")
        logger.info(f"eval dataset size, {dataset_size}")
        self.img_ids.clear()
        dataset_iter = self.dataset.create_dict_iterator(output_numpy=True)
        for idx, data in enumerate(dataset_iter):
            self.inference_step(idx, data)
            if idx % 100 == 0 and idx != 0:
                logger.info(f"inference network [{idx} / {dataset_size}]")
        logger.info(f"end inference network")
        self.save_prediction(cur_epoch, cur_step)
        result_file_path = self.file_path
        if self.eval_parallel:
            # Only support multiple devices on single machine
            self.synchronize()
            file_path = os.listdir(self.dir_path)
            result_file_path = [os.path.join(self.dir_path, path) for path in file_path]
        self.detection_engine.get_eval_result(result_file_path)

    def save_prediction(self, cur_epoch=0, cur_step=0):
        logger.info("Save bbox prediction result.")
        if self.eval_parallel:
            rank_id = self.rank_id
            self.dir_path = os.path.join(self.save_prefix, f"eval_epoch{cur_epoch}-step{cur_step}")
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path, exist_ok=True)
            file_name = f"epoch{cur_epoch}-step{cur_step}-rank{rank_id}.json"
            self.file_path = os.path.join(self.dir_path, file_name)
            if os.path.exists(self.file_path):
                import shutil

                shutil.rmtree(self.file_path, ignore_errors=True)
        else:
            t = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            self.file_path = self.save_prefix + "/predict" + t + ".json"
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.data_list, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What():{}".format(str(e)))
        else:
            self.data_list.clear()


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class COCOData:
    """Class to save COCO related variables."""

    def __init__(self, ann_path: str) -> None:
        self.ann_path: str = ann_path
        self.coco: COCO = COCO(self.ann_path)
        self.img_ids: List[int] = list(sorted(self.coco.imgs.keys()))
        self.cat_ids: List[int] = self.coco.getCatIds()


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


class DetectionEngine:
    """Detection engine"""

    def __init__(self, cfg, task="val"):
        self.cfg = cfg
        self.input_shape = None

        self.conf_thre = cfg.infer.conf_threshold
        self.nms_thre = cfg.infer.nms_threshold
        self.max_box_num = cfg.infer.max_box_num
        self.predict_box = cfg.infer.predict_box_format
        self.class_num = cfg.data.nc
        if task == "val":
            self.annFile = os.path.join(cfg.data.dataset_dir, cfg.data.val_anno_path)
        else:
            self.annFile = os.path.join(cfg.data.dataset_dir, cfg.data.test_anno_path)
        self.coco_data = COCOData(self.annFile) if os.path.exists(self.annFile) else None
        self.is_segment = cfg.data.is_segment
        self.eval_types = ["bbox"]
        if self.is_segment:
            self.eval_types.append("segm")

    def detection(self, predicts, img_shape, img_ids, masks=None):
        """
        Post process nms and detection
        Args:
            predicts: network predicts, shape is (batch_size, detections_num, 5+class_num),
                     5 is(x1, y1, x2, y2, obj_conf), xy are true size.
            img_shape: original image shape
            img_ids: image id in coco annotations json

        Returns:
            coco_format data list.
        """
        # post process nms
        predicts = self.postprocess(predicts, self.conf_thre, self.nms_thre, masks)
        return self.convert_to_coco_format(predicts, img_shape, img_ids)

    def postprocess(self, prediction, conf_thre=0.7, nms_thre=0.45, masks=None):
        """nms"""
        output = []
        for i in range(prediction.shape[0]):
            bboxes = prediction[i][..., : self.class_num * 4]
            scores = prediction[i][..., self.class_num * 4 :]
            output.append(multiclass_nms(bboxes, scores, conf_thre, nms_thre, max_num=-1, multi_masks=masks))
        return output

    def convert_to_coco_format(self, predicts, img_shapes, ids):
        """convert to coco format"""
        data_list = []
        for output, ori_img_h, ori_img_w, img_id in zip(predicts, img_shapes[:, 0], img_shapes[:, 1], ids):
            if self.is_segment:
                segs = output[1]
                output = output[0]
            if len(output) < 1:
                continue
            bboxes = output[:, 0:4]
            h, w = self.input_shape[:2]
            scale = min(h / ori_img_h, w / ori_img_w)

            bboxes = bboxes / scale
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, ori_img_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, ori_img_h)
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = self.coco_data.cat_ids[int(cls[ind])]
                segmentation = []
                if self.is_segment:
                    segmentation = self.get_seg_masks(segs[ind], bboxes[ind], ori_img_h, ori_img_w)
                    segmentation["counts"] = segmentation["counts"].decode()
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": segmentation,
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def get_seg_masks(self, mask_pred, bbox, h, w):
        """Get segmentation masks from mask_pred and bboxes"""
        mask_pred = mask_pred.astype(np.uint8)
        im_mask = np.zeros((int(h), int(w)), dtype=np.uint8)
        bbox_x, bbox_y, bbox_w, bbox_h = int(bbox[0]), int(bbox[1]), int(bbox[2] + 0.5), int(bbox[3] + 0.5)
        bbox_mask = cv2.resize(mask_pred, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
        bbox_mask = (bbox_mask > 0.5).astype(np.uint8)
        im_mask[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w] = bbox_mask

        rle = maskUtils.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        return rle

    def get_dt_list(self, file_path: List[str]):
        dt_list = []
        dt_ids_set = set([])
        logger.info(f"Total {len(file_path)} json files")
        logger.info(f"File list: {file_path}")

        for path in file_path:
            ann_list = []
            try:
                with open(path, "r") as f:
                    ann_list = json.load(f)
            except json.decoder.JSONDecodeError:
                pass  # json file is empty
            else:
                ann_ids = set(ann["image_id"] for ann in ann_list)
                diff_ids = ann_ids - dt_ids_set
                ann_list = [ann for ann in ann_list if ann["image_id"] in diff_ids]
                dt_ids_set = dt_ids_set | diff_ids
                dt_list.extend(ann_list)
        return dt_list

    def get_coco_from_dt_list(self, dt_list) -> COCO:
        cocoDt = COCO()
        cocoDt.dataset = {}
        cocoDt.dataset["images"] = [img for img in self.coco_data.coco.dataset["images"]]
        cocoDt.dataset["categories"] = copy.deepcopy(self.coco_data.coco.dataset["categories"])
        logger.info(f"Number of dt boxes: {len(dt_list)}")
        for idx, ann in enumerate(dt_list):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = idx + 1
            ann["iscrowd"] = 0
        cocoDt.dataset["annotations"] = dt_list
        cocoDt.createIndex()
        return cocoDt

    def compute_coco(self, cocoGt, cocoDt):
        for eval_type in self.eval_types:
            cocoEval = COCOeval(cocoGt, cocoDt, eval_type)
            cocoEval.evaluate()
            cocoEval.accumulate()
            rdct = Redirct()
            stdout = sys.stdout
            sys.stdout = rdct
            cocoEval.summarize()
            sys.stdout = stdout
            eval_print_str = f"\n=============coco eval {eval_type} result=========\n{rdct.content}"
            logger.info(eval_print_str)
        return rdct.content, cocoEval.stats[0]

    def get_eval_result(self, file_path: Union[str, List[str]]):
        """Get eval result"""
        if file_path is None:
            return None, None
        cocoGt = self.coco_data.coco
        if isinstance(file_path, str):
            cocoDt = cocoGt.loadRes(file_path)
        elif isinstance(file_path, list):
            dt_list = self.get_dt_list(file_path)
            cocoDt = self.get_coco_from_dt_list(dt_list)
        return self.compute_coco(cocoGt, cocoDt)
