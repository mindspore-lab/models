from mindspore import nn, ops, Tensor
import mindspore.numpy as mnp
import mindspore as ms
from model.utils import meshgrid


def make_grid(h, w, dtype=ms.float32):
    yv, xv = meshgrid((mnp.arange(h), mnp.arange(w)), indexing='ij')
    return ops.cast(ops.stack((xv, yv), 2), dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        anchor (list): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32
        scale (float): scale, default 1.

    Return:
        box (list): decoded box, [x, y, w, h], all have the shape [b, na, h, w, 1]
    """
    x, y, w, h = ops.split(box, 1, -1)
    na, grid_h, grid_w = x.shape[1:4]
    grid = make_grid(grid_h, grid_w, x.dtype).reshape((1, 1, grid_h, grid_w, 2))
    x1 = (x + grid[:, :, :, :, 0:1]) / grid_w
    y1 = (y + grid[:, :, :, :, 1:2]) / grid_h

    anchor = anchor.reshape((1, na, 1, 1, 2))
    w1 = ops.exp(w) * anchor[:, :, :, :, 0:1] / (downsample_ratio * grid_w)
    h1 = ops.exp(h) * anchor[:, :, :, :, 1:2] / (downsample_ratio * grid_h)

    return ops.concat((x1, y1, w1, h1), -1)


def xywh2xyxy(box):
    box[..., 0] = box[..., 0] - box[..., 2] * 0.5
    box[..., 1] = box[..., 1] - box[..., 3] * 0.5
    box[..., 2] = box[..., 0] + box[..., 2] * 0.5
    box[..., 3] = box[..., 1] + box[..., 3] * 0.5
    return box


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


def batch_iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = ops.maximum(px1y1, gx1y1)
    x2y2 = ops.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


class YOLOv3Loss(nn.Cell):

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=None,
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None,
                 anchors=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance
        """
        super(YOLOv3Loss, self).__init__()
        if downsample is None:
            downsample = [32, 16, 8]
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.anchors = Tensor(anchors, ms.float32)
        self.loss_item_name = ['total_loss', 'loss_xy', 'loss_wh', 'loss_obj', 'loss_cls']

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = ops.concat([gxy, gwh], axis=-1)

        iou = batch_iou_similarity(pbox, gbox)
        iou = ops.stop_gradient(iou)
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = ops.cast(iou_max <= self.ignore_thresh, pbox.dtype)
        iou_mask = ops.stop_gradient(iou_mask)

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = ops.cast(tobj > 0, pbox.dtype)
        obj_mask = ops.stop_gradient(obj_mask)

        loss_obj = ops.binary_cross_entropy_with_logits(
            pobj, obj_mask, weight=ops.ones_like(pobj), pos_weight=ops.ones_like(pobj), reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * ops.cast(tcls > 0., tcls.dtype) + neg * ops.cast(tcls <= 0., tcls.dtype)

        loss_cls = ops.binary_cross_entropy_with_logits(
            pcls, tcls, weight=ops.ones_like(pcls), pos_weight=ops.ones_like(pcls), reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1., eps=1e-10):
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj

        x = scale * ops.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * ops.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = ops.binary_cross_entropy(x, tx, reduction='none')
            loss_y = ops.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = ops.abs(x - tx)
            loss_y = ops.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum((1, 2, 3, 4)).mean()

        loss_w = ops.abs(w - tw)
        loss_h = ops.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum((1, 2, 3, 4)).mean()

        box = ops.concat((x, y, w, h), axis=-1)
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum((1, 2, 3, 4)).mean()
        return loss_xy, loss_wh, loss_obj, loss_cls

    def construct(self, input0, input1, input2, target0, target1, target2, gt_box):
        inputs = (input0, input1, input2)
        loss_xy, loss_wh, loss_obj, loss_cls = 0.0, 0.0, 0.0, 0.0
        gt_targets = (target0, target1, target2)
        i = 0
        for x, t, downsample in zip(inputs, gt_targets, self.downsample):
            loss_xy_simple, loss_wh_simple, loss_obj_simple, loss_cls_simple = self.yolov3_loss(x.astype(ms.float32), t, gt_box, self.anchors[i], downsample, self.scale_x_y)
            loss_xy += loss_xy_simple
            loss_wh += loss_wh_simple
            loss_obj += loss_obj_simple
            loss_cls += loss_cls_simple
            i += 1

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, ops.stop_gradient(ops.stack((loss, loss_xy, loss_wh, loss_obj, loss_cls)))
