import argparse
import ast
import os
import time
import cv2
import numpy as np
import time
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore_lite as mslite


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--mindir_path", type=str, help="mindir path")
    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")
    parser.add_argument("--result_folder", type=str, default="./predict_result", help="predicted results folder")
    parser.add_argument("--image_label_path", type=str, default='', help="path to image label, calculate iou accuracy")

    return parser


def load_img(data_dir):
    image_list = filter(lambda x: x.find('sat') != -1, os.listdir(data_dir))
    image_ids = list(map(lambda x: x[:-8], image_list))
    transform = transforms.Compose([
        vision.Decode(),
        vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR),
        vision.Normalize((103.53, 116.28, 123.675), (57.375, 57.120, 58.395)),
        vision.HWC2CHW()])

    for image_id in image_ids:
        img = np.fromfile(os.path.join(data_dir, '{}_sat.jpg').format(image_id), np.uint8)
        img = transform(img)
        yield image_id ,img


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        _mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[_mask].astype(int) +
            label_pred[_mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        _iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        _iou = _iou[0]
        # mean acc
        _acc = np.diag(self.hist).sum() / self.hist.sum()
        _acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        return _acc, _acc_cls, _iou


def infer(args):
    # Init
    beg_time = time.time()
    # init mslite model to predict
    context = mslite.Context()
    context.target = ["Ascend"]
    model = mslite.Model()
    print('mslite model init...')
    model.build_from_file(args.mindir_path,mslite.ModelType.MINDIR,context)
    img_count = 0
    for img_id, img in load_img(args.image_path):
        inputs = model.get_inputs()
        inputs[0].set_data_from_numpy(img)
        outputs = model.predict(inputs)
        outputs = [output.get_data_to_numpy().copy() for output in outputs]
        mask = outputs[0]
        mask[mask > 0.5] = 255
        mask[mask <= 0.5] = 0
        mask = np.squeeze(mask)
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        mask = mask.astype(np.uint8)
        cv2.imwrite(os.path.join(args.result_folder, img_id + '_mask.png'), mask)
        img_count = img_count + 1

    end_time = time.time()
    print("total time: ", end_time - beg_time, "s, img_count: ", img_count)
    pres = os.listdir(args.result_folder)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(args.image_label_path, label_name)
            pre_path = os.path.join(args.result_folder, im)
            label = cv2.imread(lab_path, 0)
            pre = cv2.imread(pre_path, 0)
            label[label > 0] = 1
            pre[pre > 0] = 1
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou = el.evaluate(predicts, labels)
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print("predict completed.")
if __name__ == "__main__":
    parser = get_parser_infer()
    args = parser.parse_args()
    infer(args)