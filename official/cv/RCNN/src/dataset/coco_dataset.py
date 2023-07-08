import os
import copy
import cv2
import numpy as np
from ..utils import logger

__all__ = ["COCODataset"]


class COCODataset:
    """
    Load the COCO dataset, parse the labels of each image to form a list of dictionaries,
    apply multi_images fusion data enhancement in __getitem__()
    COCO dataset download URL: http://cocodataset.org

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        for example:
            COCO_ROOT
                ├── annotations
                │     └── instances_train2017.json
                └── train2017
                      ├── 000000000001.jpg
                      └── 000000000002.jpg
            dataset_dir (str): ./COCO_ROOT
            image_dir (str): ./train2017
            anno_path (str): ./annotations/instances_train2017.json

        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. True as default
    """

    def __init__(
        self,
        dataset_dir="",
        image_dir="",
        anno_path="",
        load_crowd=False,
        allow_empty=True,
        empty_ratio=1.0,
        is_segmentaion=False,
        seg_size=28,
    ):
        self.dataset_dir = dataset_dir
        self.anno_path = anno_path
        self.image_dir = image_dir
        self.load_image_only = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.is_segmentaion = is_segmentaion
        self.seg_size = seg_size if isinstance(seg_size, (tuple, list)) else (seg_size, seg_size)
        assert len(self.seg_size) == 2
        self.parse_dataset()

    def __len__(
        self,
    ):
        return len(self.imgs_records)

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        record_out = copy.deepcopy(self.imgs_records[idx])
        img_path = record_out["im_file"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_segmentaion:
            return (
                img,
                record_out["im_file"],
                record_out["im_id"],
                record_out["ori_shape"],
                record_out["gt_bbox"],
                record_out["gt_class"],
                record_out["gt_poly"],
            )
        return (
            img,
            record_out["im_file"],
            record_out["im_id"],
            record_out["ori_shape"],
            record_out["gt_bbox"],
            record_out["gt_class"],
        )

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith(".json"), "invalid coco annotation file: " + anno_path
        from pycocotools.coco import COCO

        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({coco.loadCats(catid)[0]["name"]: clsid for catid, clsid in self.catid2clsid.items()})

        if "annotations" not in coco.dataset:
            self.load_image_only = True
            logger.warning(
                "Annotation file: {} does not contains ground truth "
                "and load image information only.".format(anno_path)
            )

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno["file_name"]
            im_w = float(img_anno["width"])
            im_h = float(img_anno["height"])
            im_path = os.path.join(image_dir, im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                logger.warning("Illegal image file: {}, and it will be " "ignored".format(im_path))
                continue

            if im_w < 32 or im_h < 32:
                logger.warning(
                    "Illegal width: {} or height: {} in annotation, "
                    "and im_id: {} will be ignored".format(im_w, im_h, img_id)
                )
                continue

            img_rec = {
                "im_file": im_path,
                "im_id": np.array([img_id]),
                "ori_shape": np.array([im_h, im_w]),  # for eval, [h, w]
                "pad": np.array([0.0, 0.0]),  # for eval, [padh, padw]
                "ratio": np.array([1.0, 1.0]),  # for eval, [ratio_h, ratio_w]
            }

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    # check gt bbox
                    if inst.get("ignore", False):
                        continue
                    if "bbox" not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst["bbox"])):
                            continue

                    x1, y1, box_w, box_h = inst["bbox"]
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 0.5
                    if inst["area"] > 0 and box_w >= eps and box_h >= eps:
                        inst["clean_bbox"] = [round(float(x), 3) for x in [x1, y1, x2, y2]]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            "Found an invalid bbox in annotations, drop: im_id: {}, "
                            "area: {} x1: {}, y1: {}, x2: {}, y2: {}.".format(
                                img_id, float(inst["area"]), x1, y1, x2, y2
                            )
                        )

                num_bbox = len(bboxes)
                if num_bbox > 0:
                    gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                    gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                    if self.is_segmentaion:
                        gt_poly = {"segmentations": []}

                    for i, box in enumerate(bboxes):
                        catid = box["category_id"]
                        gt_class[i][0] = self.catid2clsid[catid]
                        gt_bbox[i, :] = box["clean_bbox"]
                        if self.is_segmentaion:
                            seg = box["segmentation"]
                            gt_poly["segmentations"].append(seg)

                elif not self.allow_empty:
                    continue
                else:
                    gt_bbox = np.zeros((1, 4), dtype=np.float32)
                    gt_class = np.ones((1, 1), dtype=np.int32) * -1

                gt_rec = {
                    "gt_class": gt_class,
                    "gt_bbox": gt_bbox,  # (x1, y1, x2, y2)
                }
                if self.is_segmentaion:
                    gt_rec["gt_poly"] = gt_poly

                for k, v in gt_rec.items():
                    img_rec[k] = v
            records.append(img_rec)
        self.imgs_records = records
