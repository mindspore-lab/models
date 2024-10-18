import numpy as np
import mindspore as ms
from mindspore import ops

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, ms.Tensor)
            ), "MaskData only supports list, numpy arrays, and ms tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, ms.Tensor)
        ), "MaskData only supports list, numpy arrays, and ms tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: ms.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, ms.Tensor):
                self._stats[k] = v[ms.Tensor(keep)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.asnumpy()]
            elif isinstance(v, list) and keep.dtype == ms.bool_:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, ms.Tensor):
                self._stats[k] = ops.cat([self._stats[k], v], axis=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, ms.Tensor):
                self._stats[k] = v.asnumpy()


def is_box_near_crop_edge(
    boxes: ms.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> ms.Tensor:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_ms = ms.Tensor(crop_box, dtype=ms.float32)
    orig_box_ms = ms.Tensor(orig_box, dtype=ms.float32)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    # two nan is considered as equal. This is compat for Ascend.
    near_crop_edge = ops.isclose(boxes, crop_box_ms[None, :], atol=atol, rtol=0.0, equal_nan=True)
    near_image_edge = ops.isclose(boxes, orig_box_ms[None, :], atol=atol, rtol=0.0, equal_nan=True)
    near_crop_edge = ops.logical_and(near_crop_edge, ~near_image_edge)
    return ops.any(near_crop_edge, axis=1)


def box_xyxy_to_xywh(box_xyxy: ms.Tensor) -> ms.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_ms(tensor: ms.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(start_dim=1)

    # Compute change indices
    # diff = tensor[:, 1:] ^ tensor[:, :-1]
    diff = ms.numpy.logical_xor(tensor[:, 1:], tensor[:, :-1])
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = ops.cat(
            [
                ms.Tensor([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                ms.Tensor([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.asnumpy().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


def calculate_stability_score(
    masks: ms.Tensor, mask_threshold: float, threshold_offset: float
) -> ms.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    intersections = (masks > (mask_threshold + threshold_offset)).astype(ms.float32).sum(-1).sum(-1)
    unions = (masks > (mask_threshold - threshold_offset)).astype(ms.float32).sum(-1).sum(-1)
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: ms.Tensor, crop_box: List[int]) -> ms.Tensor:
    x0, y0, _, _ = crop_box
    offset = ms.Tensor([[x0, y0, x0, y0]])
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: ms.Tensor, crop_box: List[int]) -> ms.Tensor:
    x0, y0, _, _ = crop_box
    offset = ms.Tensor([[x0, y0]])
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(
    masks: ms.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> ms.Tensor:
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return ops.pad(masks.astype(ms.int32), pad, value=0).astype(ms.int32)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def batched_mask_to_box(masks: ms.Tensor) -> ms.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # ms.max below raises an error on empty inputs, just skip in this case
    if ops.numel(masks) == 0:
        return ops.zeros((*masks.shape[:-2], 4))

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(start_dim=0, end_dim=-3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    masks = masks.astype(ms.int32)
    in_height, _ = ops.max(masks, axis=-1)

    in_height_coords = in_height * ops.arange(h)[None, :]
    bottom_edges, _ = ops.max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + h * (1 - in_height)
    top_edges, _ = ops.min(in_height_coords, axis=-1)

    # Get left and right edges
    in_width, _ = ops.max(masks, axis=-2)
    in_width_coords = in_width * ops.arange(w)[None, :]
    right_edges, _ = ops.max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + w * (1 - in_width)
    left_edges, _ = ops.min(in_width_coords, axis=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    # empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    empty_filter = ops.logical_or((right_edges < left_edges), (bottom_edges < top_edges))
    out = ops.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: ms.Tensor, boxes2: ms.Tensor) -> ms.Tensor:
    """
    Computes the iou of two set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes1 (Tensor[N, 4]): the first set of boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        boxes2 (Tensor[M, 4]): the second set of boxes.

    Returns:
        Tensor[N, M]: the iou matrix of the two sets of boxes
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = ops.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = ops.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # [N, M]


def nms(boxes_scores, iou_threshold: float):
    boxes = boxes_scores[:, :4]
    scores = boxes_scores[:, 4]
    keep = []  # idx of boxes to keep
    idxs = scores.argsort()  # ascend
    while idxs.numel() > 0:
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index.reshape(-1))
        if idxs.shape[0] == 1:  # only one box
            break
        idxs = idxs[:-1]  # remove max score box
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # iou between max score box(1) with other boxes(many)
        idxs = idxs[ious[0] <= iou_threshold]  # discard boxes overlap too much with max score box

    keep = ops.cat(keep)
    return keep