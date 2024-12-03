import mindspore as ms
from mindspore import ops
import numpy as np
__all__ = ["can_collator",]

def can_collator(batch_image, batch_label, BatchInfo):
    """
    batch: [
        image [batch_size, channel, maxHinbatch, maxWinbatch]
        image_mask [batch_size, channel, maxHinbatch, maxWinbatch]
        label [batch_size, maxLabelLen]
        label_mask [batch_size, maxLabelLen]
        ...
    ]
    """

    batch = [list(item) for item in zip(batch_image, batch_label)]
    max_width, max_height, max_length = 0, 0, 0
    bs, channel = len(batch), batch[0][0].shape[0]
    proper_items = []
    for item in batch:
        if (
                item[0].shape[1] * max_width > 1600 * 320
                or item[0].shape[2] * max_height > 1600 * 320
        ):
            continue
        max_height = (
            item[0].shape[1] if item[0].shape[1] > max_height else max_height
        )
        max_width = (
            item[0].shape[2] if item[0].shape[2] > max_width else max_width
        )
        max_length = (
            len(item[1]) if len(item[1]) > max_length else max_length
        )
        proper_items.append(item)
    

    images = np.zeros(
        (len(proper_items), channel, max_height, max_width), dtype="float32"
    )
    image_masks = np.zeros(
        (len(proper_items), 1, max_height, max_width), dtype="float32"
    )
    labels = np.zeros(
        (len(proper_items), max_length), dtype="int32"
    )
    label_masks = np.zeros(
        (len(proper_items), max_length), dtype="int32"
    )  

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = len(proper_items[i][1])
        labels[i][:l] = proper_items[i][1]
        label_masks[i][:l] = 1

    return images, image_masks, labels, label_masks
