import os
import cv2
import numpy as np


def cal_img_pad(ori_img_shape, target_size, keep_ratio=True):
    if keep_ratio:
        im_scale = min(target_size[1] / ori_img_shape[1], target_size[0] / ori_img_shape[0])
        im_scale_x, im_scale_y = im_scale, im_scale
    else:
        im_scale_y = target_size[0] / ori_img_shape[0]
        im_scale_x = target_size[1] / ori_img_shape[1]
    pad_h = target_size[0] - int(im_scale_y * ori_img_shape[0])
    pad_w = target_size[1] - int(im_scale_x * ori_img_shape[1])
    return pad_h, pad_w


def draw_label(label, palette, ignore_label=255, label_map=None):
    label = label.astype(int)
    if label_map is not None:
        temp = label.copy()
        for k, v in label_map.items():
            label[temp == k] = v
    palette = np.array(palette).astype(np.uint8)
    label[label==ignore_label] = len(palette) - 1
    res = palette[label]
    return res


def draw_title(classes, palette, title_h, title_w):
    palette = np.array(palette).astype(np.uint8)
    title = np.ones((title_h, title_w, 3), np.uint8) * 255
    h, w = 10, 0
    for cls, pal in zip(classes, palette):
        color = np.ones((30, 80, 3), np.uint8) * pal
        title[h:h+30, w:w+80] = color
        cv2.putText(title, cls, (w+1, h+65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        w += 200
        if w + 200 > title_w:
            h += 100
            w = 0
    return title


def visualize(id, save_path, img, pred, classes, palette, ignore_label=255,
              img_shape=None, keep_ratio=True, label_map=None):
    assert img.shape[:2] == pred.shape[:2]
    assert len(classes) == len(palette)
    if img_shape is not None:
        pad_h, pad_w = cal_img_pad(img_shape, img.shape[:2], keep_ratio=keep_ratio)
        img = img[:pad_h, :pad_w]
        pred = pred[:pad_h, :pad_w]
    h, w = img.shape[:2]
    title_h = int((200 * len(classes)) / (2 * w + 30) + 0.5) * 80
    title = draw_title(classes, palette, title_h, w * 2 + 30)
    res = np.ones((h + 10 + title_h, w * 2 + 30, 3), np.uint8) * 255
    if len(img.shape) == 2:
        img = draw_label(img, palette, ignore_label, label_map)
    pred = draw_label(pred, palette, ignore_label, label_map)
    res[10:10+h, 10:10+w, :] = img
    res[10:10+h, 20+w:20+w+w, :] = pred
    res[10+h:10+h+title_h] = title
    cv2.imwrite(os.path.join(save_path, f"{id}.png"), res)


if __name__ == "__main__":
    from cityscapes import Cityscapes
    save_path = "images"
    os.makedirs(save_path, exist_ok=True)
    img = cv2.imread("")
    pred = cv2.imread("")[:, :, 0]
    classes = Cityscapes().classes
    palette = Cityscapes().palette
    label_map = Cityscapes().label_mapping
    visualize(0, save_path, img, pred, classes, palette, ignore_label=255,
              img_shape=None, keep_ratio=True, label_map=label_map)
