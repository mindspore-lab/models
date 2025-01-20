import os
from time import time
import cv2
import numpy as np

import mindspore.context as context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor

from models.dlinknet import DLinkNet34, DLinkNet50
from config import parse_args


class TTAFrame:
    def __init__(self, net):
        self.net = net()

    def load(self, path):
        param_dict = load_checkpoint(path)
        load_param_into_net(self.net, param_dict)

    def test_one_img_from_path(self, path, eval_mode=True):
        if eval_mode:
            self.net.set_train(False)

        return self.test_one_img_from_path_0(path)

    def test_one_img_from_path_0(self, path):
        img = cv2.imread(path)

        img = np.concatenate([img[None]]).transpose((0, 3, 1, 2))
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = Tensor(img)

        _mask = self.net.construct(img).squeeze().asnumpy()

        return _mask


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


if __name__ == '__main__':
    args = parse_args()

    # test_part
    trained_ckpt_path = args.trained_ckpt
    predict_path = args.predict_path
    # eval part
    label_path = args.label_path

    # test part
    # context set GRAPH_MODE PYNATIVE_MODE  Ascend
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    val = os.listdir(args.data_dir)
    if args.model_name == 'dlinknet34':
        solver = TTAFrame(DLinkNet34)
    else:
        solver = TTAFrame(DLinkNet50)
    solver.load(trained_ckpt_path)
    tic = time()
    for i, name in enumerate(val):
        if i % 10 == 0:
            print(i / 10, '    ', '%.2f' % (time() - tic))
        mask = solver.test_one_img_from_path(os.path.join(args.data_dir, name))
        mask[mask > 0.5] = 255
        mask[mask <= 0.5] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(os.path.join(predict_path, name[:-7] + 'mask.png'), mask.astype(np.uint8))

    # eval part
    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
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
