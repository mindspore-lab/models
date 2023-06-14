import mindspore.dataset as de

from .transforms_factory import create_transform
from .coco_dataset import COCODataset

__all__ = ["create_dataloader"]


def create_dataloader(
    data_config,
    task,
    per_batch_size,
    rank=0,
    rank_size=1,
    shuffle=True,
    drop_remainder=False,
    is_segmentaion=False,
    seg_size=28,
    num_parallel_worker=8,
):
    if task == "train":
        image_dir = data_config.train_img_dir
        anno_path = data_config.train_anno_path
        trans_config = getattr(data_config, "train_transforms", data_config)
        drop_remainder = True
    elif task in ("val", "eval"):
        image_dir = data_config.val_img_dir
        anno_path = data_config.val_anno_path
        trans_config = getattr(data_config, "eval_transforms", data_config)
    else:
        raise NotImplementedError
    item_transforms = getattr(trans_config, "item_transforms", [])
    transforms_name_list = []
    for transform in item_transforms:
        transforms_name_list.extend(transform.keys())
    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        transform = create_transform(item_transforms[i])
        transforms_list.append(transform)

    dataset = COCODataset(
        dataset_dir=data_config.dataset_dir,
        image_dir=image_dir,
        anno_path=anno_path,
        allow_empty=(task != "train"),
        is_segmentaion=is_segmentaion,
        seg_size=seg_size,
    )
    dataset_column_names = ["image", "im_file", "im_id", "ori_shape", "gt_bbox", "gt_class"]
    if is_segmentaion:
        dataset_column_names.append("gt_mask")

    ds = de.GeneratorDataset(
        dataset,
        column_names=dataset_column_names,
        num_parallel_workers=num_parallel_worker,
        shuffle=shuffle,
        num_shards=rank_size,
        shard_id=rank,
    )
    columns = ["image", "gt_bbox", "gt_class"]
    if is_segmentaion:
        columns.append("gt_mask")
    ds = ds.map(
        operations=transforms_list,
        input_columns=columns,
        num_parallel_workers=num_parallel_worker,
        python_multiprocessing=True,
    )

    if task == "train":
        ds = ds.project(columns)
    ds = ds.batch(per_batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_parallel_worker)

    return ds, dataset
