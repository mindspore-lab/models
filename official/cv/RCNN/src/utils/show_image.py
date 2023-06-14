import os
import cv2
import numpy as np


def save_image(image_path, bboxes, categories, name, save_path):
    img = cv2.imread(image_path)
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        categories_size = cv2.getTextSize(category + "0", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()

        if bbox[1] - categories_size[1] - 3 < 0:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] + 2),
                (bbox[0] + categories_size[0], bbox[1] + categories_size[1] + 3),
                color=color,
                thickness=-1,
            )
            cv2.putText(
                img,
                category,
                (bbox[0], bbox[1] + categories_size[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                thickness=1,
            )
        else:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] - categories_size[1] - 3),
                (bbox[0] + categories_size[0], bbox[1] - 3),
                color,
                thickness=-1,
            )
            cv2.putText(img, category, (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)
    img_dir, file_path = os.path.split(image_path)
    file_name, ex = os.path.splitext(file_path)

    cv2.imwrite(os.path.join(save_path, f"{file_name}_{name}{ex}"), img)
    return img


import json


def get_dt_list(file_path, classs_dict, coco, image_root, name="pred"):
    dt_dict = {}
    try:
        with open(file_path, "r") as f:
            ann_list = json.load(f)
    except json.decoder.JSONDecodeError:
        pass  # json file is empty
    else:
        for ann in ann_list:
            ann_id = ann["image_id"]
            if ann_id in dt_dict:
                dt_dict[ann_id].append(ann)
            else:
                dt_dict[ann_id] = [ann]
        for ann_id in dt_dict.keys():
            image_info = coco.loadImgs(ann_id)
            file_name = image_info[0]["file_name"]
            image_path = os.path.join(image_root, file_name)
            annos = []
            cls = []
            for label in dt_dict[ann_id]:
                bbox = label["bbox"]
                class_name = classs_dict[label["category_id"]] + str(label["score"])[:5]
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append(np.array([x1, y1, x2, y2]))
                cls.append(class_name)
            save_image(image_path, annos, cls, name, save_path)


def read_from_json(coco, image_root, save_path, name):
    """Get image path and annotation from COCO."""
    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(image_root, file_name)
        annos = []
        cls = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            x1, x2 = bbox[0], bbox[0] + bbox[2]
            y1, y2 = bbox[1], bbox[1] + bbox[3]
            annos.append(np.array([x1, y1, x2, y2]))
            cls.append(class_name)
        save_image(image_path, annos, cls, name, save_path)

    return image_files, image_anno_dict


anno_json = "/data1/mindspore_dataset/coco2017/annotations/instances_val2017.json"
image_root = "/data1/mindspore_dataset/coco2017/val2017"
save_path = "./predict"
pred_json = "output/"
os.makedirs(save_path, exist_ok=True)
classs_dict = {}
from pycocotools.coco import COCO

coco = COCO(anno_json)
cat_ids = coco.loadCats(coco.getCatIds())
for cat in cat_ids:
    classs_dict[cat["id"]] = cat["name"]

# read_from_json(coco, image_root, save_path, "gt")
get_dt_list(pred_json, classs_dict, coco, image_root, "pred")
