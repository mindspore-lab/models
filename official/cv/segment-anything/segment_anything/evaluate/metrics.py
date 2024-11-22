from typing import List

import mindspore as ms

from segment_anything.utils.registry import METRIC_REGISTRY
from segment_anything.utils.utils import calc_iou, all_reduce, reduce_with_mask


@METRIC_REGISTRY.registry_module()
class MaskMiou(ms.train.Metric):
    def __init__(self):
        super(MaskMiou, self).__init__()
        self.sum_iou = ms.Tensor(0.0)
        self.num_step = ms.Tensor(0.0)
        self.clear()

    def clear(self):
        self.sum_iou = ms.Tensor(0.0)
        self.num_step = ms.Tensor(0.0)

    def update(self, *inputs):
        """
        update metrics of a batch
        """
        pred, gt = inputs
        pred_mask = pred['masks'].astype(ms.float32)  # bool -> float32
        gt_mask = gt['masks'].astype(ms.float32)
        valid_boxes = gt.get('valid_boxes')

        assert pred_mask.shape == gt_mask.shape
        iou = calc_iou(pred_mask, gt_mask)  # (b, n)
        iou_per_batch = reduce_with_mask(iou, valid_boxes, reduction='sum')  # (1,)
        valid_per_batch = ms.ops.sum(valid_boxes.astype(iou.dtype))
        self.num_step += valid_per_batch
        self.sum_iou += iou_per_batch
    
    def eval(self):
        # reduce from all the devices
        sum_iou = all_reduce(self.sum_iou)
        num_step = all_reduce(self.num_step)
        miou = sum_iou / num_step
        print(sum_iou, num_step, miou)
        miou = miou.asnumpy()
        res_dict = dict(miou=miou)
        return res_dict


@METRIC_REGISTRY.registry_module()
class MaskVisualization(ms.train.Metric):
    def __init__(self, save_dir='./vis/'):
        """
        Apply mask visualization, this is a hack implementation
        """
        super(MaskVisualization, self).__init__()
        self.save_dir = save_dir

    def clear(self):
        pass

    def update(self, **inputs):
        """
        save image with prediction and gt to directory
        """
        pred, gt = inputs
        pred_mask = pred['mask']
        gt_mask = gt['mask']
        valid_box = gt['valid_boxes']
        origin_hw = gt['origin_hw']
        # TODO save picture here
        pass


    def eval(self):
        pass


def create_metric(metric: List):
    """
    instantiate metric class
    """
    metric_list = []
    for m in metric:
        metric_list.append(METRIC_REGISTRY.instantiate(**m))
    return metric_list
