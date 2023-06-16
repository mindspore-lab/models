import numpy as np


def nms(xyxys, scores, threshold):
    """Calculate NMS"""
    x1 = xyxys[:, 0]
    y1 = xyxys[:, 1]
    x2 = xyxys[:, 2]
    y2 = xyxys[:, 3]
    scores = scores
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, min_x2 - max_x1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1)
        intersect_area = intersect_w * intersect_h

        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area + 1e-6)
        indexes = np.where(ovr <= threshold)[0]
        order = order[indexes + 1]
    return np.array(reserved_boxes)


def batched_nms(bboxes, scores, labels, threshold):
    r"""Performs non-maximum suppression in a batched fashion.

    Args:
        bboxes (numpy.array): boxes in shape (N, 4), 4 is x1y1x2y2.
        scores (numpy.array): scores in shape (N, ).
        labels (numpy.array): labels in shape (N, ).
        threshold (float): nms threshold.

    Returns:
        - keep (numpy.array): The indices of remaining boxes in input
          boxes.
    """

    max_coordinate = bboxes.max()
    offsets = labels * (max_coordinate + 1)
    boxes_for_nms = bboxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, threshold)
    return keep


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_thr, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (numpy.array): shape (n, 4 * num_classes)
        multi_scores (numpy.array): shape (n, num_classes + 1).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): nms threshold.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.

    Returns:
        tuple: (dets, labels), dets shape (k, 5), labels shape (k). Dets are boxes with scores.
    """
    # exclude background category
    num_classes = multi_scores.shape[1] - 1
    bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
    assert num_classes == bboxes.shape[1]

    scores = multi_scores[:, :-1]

    labels = np.arange(num_classes)
    labels = np.tile(labels.reshape(1, -1), (scores.shape[0], 1))

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr
    if valid_mask.sum() > 0:
        bboxes, scores, labels = bboxes[valid_mask], scores[valid_mask], labels[valid_mask]
        keep = batched_nms(bboxes, scores, labels, nms_thr)

        if max_num > 0:
            keep = keep[:max_num]
        cls_scores = scores[:, None]
        predicts = np.concatenate((bboxes, np.ones_like(cls_scores), cls_scores, labels[:, None]), axis=-1)
        return predicts[keep]
    else:
        return []
