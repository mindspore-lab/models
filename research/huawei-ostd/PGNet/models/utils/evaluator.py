import logging
from typing import List

from tqdm import tqdm

import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

__all__ = ["Evaluator"]
_logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        network,
        dataloader,
        loss_fn=None,
        postprocessor=None,
        metrics=None,
        pred_cast_fp32=False,
        loader_output_columns=None,
        input_indices=None,
        label_indices=None,
        meta_data_indices=None,
        num_epochs=-1,
        visualize=False,
        verbose=False,
        **kwargs,
    ):
        self.net = network
        self.postprocessor = postprocessor
        self.metrics = metrics if isinstance(metrics, List) else [metrics]
        self.metric_names = []
        for m in metrics:
            assert hasattr(m, "metric_names") and isinstance(m.metric_names, List), (
                f"Metric object must contain `metric_names` attribute to indicate the metric names as a List type, "
                f"but not found in {m.__class__.__name__}"
            )
            self.metric_names += m.metric_names

        self.pred_cast_fp32 = pred_cast_fp32
        self.visualize = visualize
        self.verbose = verbose
        eval_loss = False
        if loss_fn is not None:
            eval_loss = True
            self.loss_fn = loss_fn
        assert not eval_loss, "not impl"

        # create iterator
        self.reload(
            dataloader,
            loader_output_columns,
            input_indices,
            label_indices,
            meta_data_indices,
            num_epochs,
        )

    def reload(
        self,
        dataloader,
        loader_output_columns=None,
        input_indices=None,
        label_indices=None,
        meta_data_indices=None,
        num_epochs=-1,
    ):
        # create iterator
        self.iterator = dataloader.create_tuple_iterator(num_epochs=num_epochs, output_numpy=False, do_copy=False)
        self.num_batches_eval = dataloader.get_dataset_size()

        # dataset output columns
        self.loader_output_columns = loader_output_columns or []
        self.input_indices = input_indices
        self.label_indices = label_indices
        self.meta_data_indices = meta_data_indices

    def eval(self):
        """
        Args:
        """
        eval_res = {}

        self.net.set_train(False)
        for m in self.metrics:
            m.clear()

        for i, data in tqdm(enumerate(self.iterator), total=self.num_batches_eval):
            if self.input_indices is not None:
                inputs = [data[x] for x in self.input_indices]
            else:
                inputs = [data[0]]

            if self.label_indices is not None:
                gt = [data[x] for x in self.label_indices]
            else:
                gt = data[1:]

            preds = self.net(*inputs)

            if self.pred_cast_fp32:
                if isinstance(preds, ms.Tensor):
                    preds = F.cast(preds, mstype.float32)
                else:
                    preds = [F.cast(p, mstype.float32) for p in preds]

            data_info = {"labels": gt, "img_shape": inputs[0].shape}

            if self.postprocessor is not None:
                # additional info such as image path, original image size, pad shape, extracted in data processing
                if self.meta_data_indices is not None:
                    meta_info = [data[x] for x in self.meta_data_indices]
                else:
                    # assume the indices not in input_indices or label_indices are all meta_data_indices
                    input_indices = set(self.input_indices) if self.input_indices is not None else {0}
                    label_indices = (
                        set(self.label_indices) if self.label_indices is not None else set(range(1, len(data), 1))
                    )
                    meta_data_indices = sorted(set(range(len(data))) - input_indices - label_indices)
                    meta_info = [data[x] for x in meta_data_indices]

                data_info["meta_info"] = meta_info

                # NOTES: add more if new postprocess modules need new keys. shape_list is commonly needed by detection
                possible_keys_for_postprocess = ["shape_list", "raw_img_shape"]
                for k in possible_keys_for_postprocess:
                    if k in self.loader_output_columns:
                        data_info[k] = data[self.loader_output_columns.index(k)]

                preds = self.postprocessor(preds, **data_info)

            # metric internal update
            for m in self.metrics:
                m.update(preds, gt)

            if self.verbose:
                _logger.info(f"Data meta info: {data_info}")

        for m in self.metrics:
            res_dict = m.eval()
            eval_res.update(res_dict)

        self.net.set_train(True)

        return eval_res
