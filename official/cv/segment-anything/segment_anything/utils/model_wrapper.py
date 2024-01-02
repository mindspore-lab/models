import time
from typing import Union, List

import mindspore as ms
import numpy as np
from mindspore import nn, ops, RunContext, value_and_grad, Tensor
from mindspore.dataset.engine.datasets import _set_training_dataset
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._ps_context import _enable_distributed_mindrt, _is_role_pserver, _cache_enable
from mindspore.parallel._recovery_context import _get_recovery_context
from mindspore.parallel._utils import _reset_op_id_with_offset
from mindspore.train.model import _transfer_tensor_to_tuple
from mindspore.train.dataset_helper import DatasetHelper, connect_network_with_dataset

_grad_scale = ops.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class NetWithLossWrapper(nn.Cell):
    """
    A universal wrapper for any network with any loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function
        input_indices: contains two element as follows:
            model_input_indices: The indices of the data tuples which will be fed into the network.
            label_indices: The indices of the data tuples which will be fed into the loss function.
    """

    def __init__(self,
                 net: nn.Cell,
                 loss_fn: nn.Cell,
                 input_columns: List[List[str]],
                 all_columns: List[str],
                 ):
        super().__init__(auto_prefix=False)
        self.net = net
        self.loss_fn = loss_fn
        assert len(input_columns) == 2
        self.input_indices = key2index(input_columns[0], all_columns)
        self.label_indices = key2index(input_columns[1], all_columns)
        self.all_columns = all_columns
        self.input_names = [all_columns[i] for i in self.input_indices]
        self.label_names = [all_columns[i] for i in self.label_indices]

    def construct(self, *args):
        """
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            loss_val (Tensor): loss value
        """

        # This is a very ugly workaround due to mindspore's disability of dict setitem and unpacking
        # pred = self.net(**select_inputs_by_indices(args, self.input_indices, self.all_columns, return_type='dict'))
        if len(self.input_names) == 2 and self.input_names[0] == 'image' and self.input_names[1] == 'image_patches':  # text-prompt
            pred = self.net(image=args[self.input_indices[0]], image_patches=args[self.input_indices[1]])
        elif len(self.input_names) == 2 and self.input_names[0] == 'image' and self.input_names[1] == 'boxes':
            pred = self.net(image=args[self.input_indices[0]], boxes=args[self.input_indices[1]])
        else:
            raise NotImplementedError

        if not isinstance(pred, tuple):
            pred = (pred, )
        loss_val = self.loss_fn(*pred, *select_inputs_by_indices(args, self.label_indices))

        # currently nn.TrainOneStepCell does not support loss as a tuple
        if isinstance(loss_val, tuple):
            loss_val = loss_val[0]

        return loss_val


def select_inputs_by_indices(inputs, indices, columns_names=None, return_type='tuple'):
    if return_type == 'dict':
        assert columns_names is not None
        assert len(inputs) == len(columns_names)
        new_inputs = dict()
        for i in indices:
            new_inputs[columns_names[i]] = inputs[i]
    elif return_type == 'tuple':
        new_inputs = list()
        for x in indices:
            new_inputs.append(inputs[x])
    else:
        raise NotImplementedError

    return new_inputs

def key2index(keys: Union[List[str], str], all_keys: List[str])-> Union[List[int], int]:
    """
    map keys to index of a list. Usually used in converting model and loss input str to index.
    """
    is_single_element = False
    if isinstance(keys, str):
        is_single_element = True
        keys = [keys]

    index = []
    for k in keys:
        assert k in all_keys
        ind = all_keys.index(k)
        index.append(ind)

    if is_single_element:
        index = index[0]

    return index


class TrainOneStepCellWrapper(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_sense=1,
                 grad_clip=False,
                 clip_value=0.1,
                 drop_overflow_update=True,  # do not update when overflow
                 ema=None,
                 ):
        if isinstance(scale_sense, (int, float)):
            scale_sense = nn.FixedLossScaleUpdateCell(scale_sense)
        super(TrainOneStepCellWrapper, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip if grad_clip is not None else False
        self.grad_clip_value = clip_value
        self.drop_overflow_update = drop_overflow_update if drop_overflow_update is not None else True
        self.ema = ema

    def construct(self, *inputs):
        weights = self.weights

        loss = self.network(*inputs)  # loss_items may contain both loss and auxiliary value

        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        # loss*grad_scale -> backpropagation -> grad/grad_scale
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)

        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.grad_clip_value)

        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)

        overflow = self.process_loss_scale(cond)

        if overflow:
            print(f'gradient overflow')

        if (not self.drop_overflow_update) or (not overflow):
            self.optimizer(grads)
            if self.ema is not None:
                self.ema.ema_update()

        return loss


class SamIterativeSegModel(ms.Model):
    """
    Model specialized for iterative interactive segmentation.
    """
    def _build_train_network(self):
        train_one_step_loss_cell = self._network  # train_one_step_loss_cell
        net_with_loss = train_one_step_loss_cell.network
        optimizer = train_one_step_loss_cell.optimizer
        grad_reducer = train_one_step_loss_cell.grad_reducer
        net = net_with_loss.net  # Net without loss
        loss_fn = net_with_loss.loss_fn
        weights = train_one_step_loss_cell.weights

        # for training only
        net.set_train(True)
        @ms.jit
        def forward_point(image, points=None, boxes=None, masks=None,
                          gt_mask=None, valid_boxes=None,
                          multimask_output=False, output_best_mask=True, return_low_res_mask=True):
            _pred_mask, _pred_iou, _low_res_mask = net(image, points=points, boxes=boxes, masks=masks,
                                                       multimask_output=multimask_output, output_best_mask=output_best_mask, return_low_res_mask=return_low_res_mask)
            _loss = loss_fn(_pred_mask, _pred_iou, gt_mask=gt_mask, valid_boxes=valid_boxes)
            return _loss[0], (_pred_mask, _pred_iou, _low_res_mask)

        def _train_fn(*data_element):

            # tuple to dict
            input_dict = select_inputs_by_indices(data_element, net_with_loss.input_indices, net_with_loss.all_columns, return_type='dict')
            gt_dict = select_inputs_by_indices(data_element, net_with_loss.label_indices, net_with_loss.all_columns, return_type='dict')

            grad_fn = value_and_grad(forward_point, grad_position=None, weights=weights, has_aux=True)

            # 11 iteration
            # mask_only_iter = [10, np.random.randint(1, 10)]  # the last and one random iteration
            mask_only_iter = []  # the last and one random iteration
            previous_mask = None
            previous_low_mask = None
            loss_list = []
            grad_list = []

            for i in range(2):
                s0 = time.time()
                print(f'\nstart iter {i}')
                return_pad_point = i in mask_only_iter  # for mask only iter, give a pad-point to keep static shape
                multimask_output = False
                if i == 0:
                    return_pad_point = False  # the first iteration needs a valid point
                    multimask_output = True  # the first iteration needs multi-mask output due to point ambiguity

                point_and_label = self.get_next_point(gt_dict['masks'], pred_mask=previous_mask,
                                                      return_default=return_pad_point)
                s1 = time.time()
                print(f'get next takes: {s1-s0:.2f}s')
                (loss, (mask, iou, low_res_mask)), grads = grad_fn(
                                                input_dict['image'],
                                                point_and_label,
                                                None,  # box
                                                previous_low_mask,
                                                gt_dict['masks'],
                                                gt_dict['valid_boxes'],
                                                multimask_output)
                s2 = time.time()
                print(f'f and b takes: {s2-s1:.2f}s')
                previous_mask = ops.stop_gradient(mask > loss_fn.mask_threshold)  #  (b, n, h, w)
                previous_low_mask = ops.stop_gradient(low_res_mask.expand_dims(2))  # (b, n, h, w) -> (b, n, 1, h, w)
                print(f'mask input shape {previous_low_mask.shape}')
                s3 = time.time()
                print(f'postprocess takes: {s3 - s2:.2f}s')
                grad_list.append(grads)  # grad is a tuple with Tensor element

                loss_list.append(loss)

            grad_accum = tuple([sum(k) for k in zip(*grad_list)])

            print(f'loss list', loss_list)
            t0 = time.time()
            grad_accum = grad_reducer(grad_accum)
            optimizer(grad_accum)
            t1 = time.time()
            print(f'optimize takes: {t1 - t0:.2f}s\n\n\n')

            return loss_list[0]
        return _train_fn

    def get_next_point(self, gt_mask, pred_mask=None, return_default=False):
        """
        get the next point according to the difference area of pred_mask and gt_mask
        """
        if isinstance(gt_mask, ms.Tensor):
            gt_mask = gt_mask.asnumpy()
        if isinstance(pred_mask, ms.Tensor):
            pred_mask = pred_mask.asnumpy()

        bs, n, h, w = gt_mask.shape
        if return_default:
            points = ops.zeros((bs, n, 1, 2), dtype=ms.float32)  # (bs, bs_prompt, num_point_per_batch, 2)
            labels = (-1 * ops.ones((bs, n, 1), dtype=ms.int32))
            return points, labels
        # if no pred_mask provided, sample the positive area of gt_mask
        # else sample the difference area
        triple_map = gt_mask.astype(np.int32) - pred_mask.astype(np.int32) if pred_mask is not None else gt_mask.astype(np.int32)
        # triple_map = gt_mask.astype(ms.int32)
        triple_map = triple_map.reshape((bs*n, h, w))

        points = []
        labels = []
        for i in range(bs*n):
            non_zero_ind = np.transpose(np.nonzero(triple_map[i]))  # (nz, 2)
            if len(non_zero_ind) !=0:
                # rand = np.random.randint(0, len(non_zero_ind)) # (1,)
                rand = 0
                point = non_zero_ind[rand]  # (2,)
                label = triple_map[i][point[0], point[1]]  # 1 or -1
                label = (label > 0).astype(np.int32)  # 1 or 0
            else:
                point = np.array([0, 0], dtype=np.float32)
                label = np.array(-1, dtype=np.int32)
            points.append(point[::-1].astype(np.float32))  # input point should be in (w, h) format
            labels.append(label)

        points = np.stack(points).reshape((bs, n, 1, 2))  # (bs, bs_prompt, num_point_per_batch, 2)
        labels = np.stack(labels).reshape((bs, n, 1))  # (bs, bs_prompt, num_point_per_batch)

        points = ms.Tensor(points, dtype=ms.float32)
        labels = ms.Tensor(labels, dtype=ms.int32)

        # if debug:
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     from segment_anything.utils.visualize import show_mask, show_points, show_box
        #     i = 0
        #     plt.imshow(gt_mask[0, i].asnumpy())
        #     show_points(points[0, i, 0].asnumpy(), labels[0, i, 0].asnumpy(), plt.gca())
        #     plt.show()

        return points, labels

    def get_next_point_ms(self, gt_mask, pred_mask=None, return_default=False):
        """
        get the next point according to the difference area of pred_mask and gt_mask.
        Mindspore version, slower than numpy version
        """

        bs, n, h, w = gt_mask.shape
        if return_default:
            points = ops.zeros((bs, n, 1, 2))  # (bs, bs_prompt, num_point_per_batch, 2)
            labels = (-1 * ops.ones((bs, n, 1), dtype=ms.int32))
            return points, labels
        # if no pred_mask provided, sample the positive area of gt_mask
        # else sample the difference area
        triple_map = gt_mask.astype(ms.int32) - pred_mask.astype(ms.int32) if pred_mask is not None else gt_mask.astype(ms.int32)
        # triple_map = gt_mask.astype(ms.int32)
        triple_map = triple_map.reshape(bs*n, h, w)

        points = []
        labels = []
        for i in range(bs*n):
            non_zero_ind = ops.nonzero(triple_map[i])  # (nz, 2)
            if len(non_zero_ind) !=0:
                # rand = ops.randint(0, len(non_zero_ind), (1,))[0] # (1,)
                rand = 0
                point = non_zero_ind[rand]  # (2,)
                label = triple_map[i][point[0], point[1]]  # 1 or -1
                label = (label > 0).astype(ms.int32)  # 1 or 0
            else:
                point = ms.Tensor([0, 0], dtype=ms.float32)
                label = ms.Tensor(-1, dtype=ms.int32)
            points.append(point[::-1].astype(ms.float32))  # input point should be in (w, h) format
            labels.append(label)

        points = ops.stack(points).reshape(bs, n, 1, -1)  # (bs, bs_prompt, num_point_per_batch, 2)
        labels = ops.stack(labels).reshape(bs, n, 1)  # (bs, bs_prompt, num_point_per_batch)

        # if debug:
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     from segment_anything.utils.visualize import show_mask, show_points, show_box
        #     i = 0
        #     plt.imshow(gt_mask[0, i].asnumpy())
        #     show_points(points[0, i, 0].asnumpy(), labels[0, i, 0].asnumpy(), plt.gca())
        #     plt.show()

        return points, labels

    # ---- below are overidded method to prevent incompatible runtime errors ---- #
    def _exec_preprocess(self, is_train, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1, dataset_helper=None):
        """Initializes dataset."""
        if is_train:
            network = self._train_network
            phase = 'train'
        else:
            network = self._eval_network
            phase = 'eval'

        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1

        if dataset_helper is None:
            dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

        if dataset_sink_mode:
            network = connect_network_with_dataset(network, dataset_helper)

        if _get_recovery_context("enable_recovery") and is_train:
            _set_training_dataset(dataset_helper)


        # network.set_train(is_train)
        # network.phase = phase
        self._backbone_is_train = is_train

        return dataset_helper, network

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        params = cb_params.network.get_parameters()
        for param in params:
            if param.cache_enable:
                Tensor(param).flush_from_cache()
