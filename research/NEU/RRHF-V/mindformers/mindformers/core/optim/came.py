# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""came optimizer"""
from __future__ import absolute_import

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.log import logging
from mindspore.common.initializer import initializer
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
try:
    from mindspore._checkparam import Validator as validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as validator
    import mindspore._checkparam as Rel
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['Came']


def _rms(update_tensor):
    """calculate rms"""
    return F.sqrt(P.ReduceMean(False)(F.square(update_tensor)))


def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    """Approximation of exponential moving average of square of gradient"""
    reduce_mean = P.ReduceMean(keep_dims=True)(exp_avg_sq_row, -1)
    div_val = 1.0 / P.Sqrt()(P.RealDiv()(exp_avg_sq_row, reduce_mean))
    r_factor = (P.ExpandDims()(div_val, -1))

    exp_avg_sq_col = P.ExpandDims()(exp_avg_sq_col, -2)
    c_factor = 1.0 / P.Sqrt()(exp_avg_sq_col)
    return P.Mul()(r_factor, c_factor)


reduce_mean_keep_alive = P.ReduceMean().add_prim_attr("keep_alive", True)
_came_opt = C.MultitypeFuncGraph("came_opt")


@_came_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool", "Bool", "Bool",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_opt_with_one_number(eps, clip_threshold, beta1, beta2t, beta3, weight_decay, scale_parameter,
                             compression, use_first_moment, weight_decay_flag, learning_rate,
                             grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq,
                             exp_avg_insta_row, exp_avg_insta_col):
    """Apply came optimizer to the weight parameter using Tensor."""
    cast = P.Cast()
    grad_dtype = F.dtype(grad)
    grad_shape = F.shape(grad)

    if grad_dtype == mstype.float16:
        grad = cast(grad, mstype.float32)
    p_data_fp32 = param
    if F.dtype(p_data_fp32) == mstype.float16:
        p_data_fp32 = cast(p_data_fp32, mstype.float32)

    factored = len(grad_shape) >= 2

    if scale_parameter:
        rms = _rms(p_data_fp32)
        param_scale = P.Maximum()(eps[1], rms)
        learning_rate_update = learning_rate * param_scale * F.ones_like(rms)
    else:
        learning_rate_update = learning_rate

    update = (grad ** 2) + eps[0]

    if factored:
        exp_avg_sq_row_update = cast(exp_avg_sq_row, grad_dtype)
        exp_avg_sq_row_update = P.Mul()(exp_avg_sq_row_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -1) * (1.0 - beta2t)
        exp_avg_sq_row_update = P.Add()(exp_avg_sq_row_update, update_mean)
        F.assign(exp_avg_sq_row, cast(exp_avg_sq_row_update, F.dtype(exp_avg_sq_row)))
        exp_avg_sq_row_update = exp_avg_sq_row

        exp_avg_sq_col_update = cast(exp_avg_sq_col, grad_dtype)
        exp_avg_sq_col_update = P.Mul()(exp_avg_sq_col_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -2) * (1.0 - beta2t)
        exp_avg_sq_col_update = P.Add()(exp_avg_sq_col_update, update_mean)
        F.assign(exp_avg_sq_col, cast(exp_avg_sq_col_update, F.dtype(exp_avg_sq_col)))
        exp_avg_sq_col_update = exp_avg_sq_col
        update = _approx_sq_grad(exp_avg_sq_row_update, exp_avg_sq_col_update)
        update = P.Mul()(update, grad)

    else:
        exp_avg_sq_update = cast(exp_avg_sq, grad_dtype)
        update = update * (1.0 - beta2t)
        exp_avg_sq_update = P.Add()(P.Mul()(exp_avg_sq_update, beta2t), update)
        F.assign(exp_avg_sq, cast(exp_avg_sq_update, F.dtype(exp_avg_sq)))
        exp_avg_sq_update = exp_avg_sq
        exp_avg_sq_update = 1.0 / P.Sqrt()(exp_avg_sq_update)
        update = P.Mul()(exp_avg_sq_update, grad)

    update_rms_thres = _rms(update) / clip_threshold
    update_coff = P.Maximum()(update_rms_thres, P.OnesLike()(update_rms_thres))
    update = P.RealDiv()(update, update_coff)

    if use_first_moment:
        exp_avg_update = exp_avg
        if compression:
            exp_avg_update = cast(exp_avg, grad_dtype)
        exp_avg_update = P.Add()(P.Mul()(exp_avg_update, beta1), update * (1 - beta1))
        F.assign(exp_avg, cast(exp_avg_update, F.dtype(exp_avg)))

    ###
    # CAME  optimizer modification is here
    instability_matrix = (update - exp_avg) ** 2 + eps[2]

    if factored:
        exp_avg_insta_row_update = cast(exp_avg_insta_row, grad_dtype)
        exp_avg_insta_row_update = P.Mul()(exp_avg_insta_row_update, beta3)
        update_mean = reduce_mean_keep_alive(instability_matrix, -1) * (1.0 - beta3)
        exp_avg_insta_row_update = P.Add()(exp_avg_insta_row_update, update_mean)
        F.assign(exp_avg_insta_row, cast(exp_avg_insta_row_update, F.dtype(exp_avg_insta_row)))
        exp_avg_insta_row_update = exp_avg_insta_row

        exp_avg_insta_col_update = cast(exp_avg_insta_col, grad_dtype)
        exp_avg_insta_col_update = P.Mul()(exp_avg_insta_col_update, beta3)
        update_mean = reduce_mean_keep_alive(instability_matrix, -2) * (1.0 - beta3)
        exp_avg_insta_col_update = P.Add()(exp_avg_insta_col_update, update_mean)
        F.assign(exp_avg_insta_col, cast(exp_avg_insta_col_update, F.dtype(exp_avg_insta_col)))
        exp_avg_insta_col_update = exp_avg_insta_col

        s_t = _approx_sq_grad(exp_avg_insta_row_update, exp_avg_insta_col_update)
        update = s_t * exp_avg * learning_rate_update
    else:
        update = exp_avg * learning_rate_update
    # ###

    if weight_decay_flag:
        p_data_fp32_coff = p_data_fp32 * -weight_decay * learning_rate_update
        p_data_fp32 = P.Add()(p_data_fp32, p_data_fp32_coff)
    p_data_fp32 = P.Sub()(p_data_fp32, update)
    P.Assign()(param, cast(p_data_fp32, F.dtype(param)))
    return True


@_came_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor")
def _run_fused_ada_factor(fused_ada_factor, eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                          grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq):
    fused_ada_factor(eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                     grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq)
    return True


def trans_to_tensor(param, is_tuple=False, fp32=True):
    """
    Transform params to tensor.
    """
    if param is None or isinstance(param, bool):
        return param
    data_type = mstype.float32 if fp32 else mstype.float16
    if is_tuple:
        new_param = [Tensor(ele, data_type) for ele in param]
        return tuple(new_param)
    return Tensor(param, data_type)


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class Came(Optimizer):
    r"""
    Updates gradients by the Confidence-guided Adaptive Memory Efficient Optimization (Came) algorithm.

    The Came algorithm is proposed in `CAME: Confidence-guided Adaptive Memory Efficient Optimization
    <https://arxiv.org/abs/2307.02047>`_ .

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`.
        learning_rate (Union[float, Tensor]): A value or a graph for the learning rate.
            When the learning_rate is a Tensor in a 1D dimension.
            If the type of `learning_rate` is int, it will be converted to float. Default: None.
        eps (Union[list, tuple]): The regularization constans for square gradient, parameter scale and
            instability_matrix respectively. default: (1e-30, 1e-3, 1e-16)
        clip_threshold (float): The threshold of root mean square of final gradient update. default: 1.0
        decay_rate (float): The coefficient used to compute running averages of square gradient.
            Should be in range [0.0, 1.0]. default: 0.8.
        beta1 (float): The coefficient to computing running averages of gradient. Should be in range [0.0, 1.0].
               Default: 0.9.
        beta3 (float): The coefficient to computing running averages of gradient. Should be in range [0.0, 1.0].
               Default: 0.99.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0.
            Should be in range [0.0, 1.0]. default: 0.0.
        scale_parameter (bool): If True, learning rate is scaled by root mean square of parameter. default: True
        relative_step (bool): If True, time-dependent learning rate is computed instead of external learning rate.
            default: True
        warmup_init (bool): The time-dependent learning rate computation depends on whether warm-up
            initialization is being used. default: False
        compression (bool): If True, the data type of the running averages exponent will be compression to float16.
            default: False
        loss_scale (int): An integer point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `decay_rate`, `weight_decay`, `beta1`, `beta3`, `eps` or `loss_scale` is not a float.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `decay_rate`, `weight_decay`, `beta1` or `beta3` is not in range [0.0, 1.0].

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindformers import AutoModel
        >>> from mindformers.core.optim import Came
        >>>
        >>> ms.set_context(mode=ms.context.GRAPH_MODE)
        >>> net = AutoModel.from_pretrained("llama2_7b", num_layers=2)
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = Came(params=net.trainable_params(), learning_rate=0.1)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> layernorm_params = list(filter(lambda x: 'norm' in x.name, net.trainable_params()))
        >>> no_layernorm_params = list(filter(lambda x: 'norm' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': layernorm_params, 'weight_decay': 0.01},
        ...                 {'params': no_layernorm_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = Came(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The layernorm_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_layernorm_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """
    _support_parallel_optimizer = True

    @opt_init_args_register
    def __init__(self,
                 params,
                 learning_rate=None,
                 eps=(1e-30, 1e-3, 1e-16),
                 clip_threshold=1.0,
                 decay_rate=0.8,
                 beta1=0.9,
                 beta3=0.99,
                 weight_decay=0.0,
                 scale_parameter=False,
                 relative_step=False,
                 warmup_init=False,
                 compression=False,
                 loss_scale=1):
        if compression:
            raise ValueError(f"Currently, came only supports compression equals False, but got {compression}")
        if learning_rate is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options", learning_rate)
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        if learning_rate is None and not relative_step:
            raise ValueError("Cannot learning_rate is None and relative_step=False")
        if learning_rate is None:
            learning_rate = 0.0
        if beta1 is None:
            beta1 = 0.0

        if not isinstance(learning_rate, (float, int)) and learning_rate is not None:
            if relative_step or scale_parameter:
                logging.warning("When learning_rate is learning scheduler, it not support update learning rate!")

        validator.check_value_type("loss_scale", loss_scale, [int], self.cls_name)
        super(Came, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("eps", eps, [list, tuple], self.cls_name)
        if len(eps) != 3:
            raise ValueError("eps must have 3 value: (eps1, eps2, eps3).")
        for i, ele in enumerate(eps):
            validator.check_value_type("eps{}".format(i), ele, [float], self.cls_name)
            validator.check_non_negative_float(ele, "eps{}".format(i), self.cls_name)
        validator.check_value_type("clip_threshold", clip_threshold, [float], self.cls_name)
        validator.check_non_negative_float(clip_threshold, "clip_threshold", self.cls_name)
        validator.check_value_type("decay_rate", decay_rate, [float], self.cls_name)
        validator.check_float_range(decay_rate, 0, 1, Rel.INC_BOTH, "decay_rate", self.cls_name)
        validator.check_value_type("weight_decay", weight_decay, [float], self.cls_name)
        validator.check_float_range(weight_decay, 0, 1, Rel.INC_BOTH, "weight_decay", self.cls_name)
        validator.check_value_type("scale_parameter", scale_parameter, [bool], self.cls_name)
        validator.check_value_type("relative_step", relative_step, [bool], self.cls_name)
        validator.check_value_type("warmup_init", warmup_init, [bool], self.cls_name)
        validator.check_value_type("compression", compression, [bool], self.cls_name)
        validator.check_value_type("beta1", beta1, [float], self.cls_name)
        validator.check_float_range(beta1, 0, 1, Rel.INC_BOTH, "beta1", self.cls_name)
        validator.check_value_type("beta3", beta3, [float], self.cls_name)
        validator.check_float_range(beta3, 0, 1, Rel.INC_BOTH, "beta3", self.cls_name)
        self.eps = trans_to_tensor(eps)
        self.clip_threshold = trans_to_tensor(clip_threshold)
        self.decay_rate = trans_to_tensor(-decay_rate)
        self.beta1 = trans_to_tensor(beta1)
        self.beta3 = trans_to_tensor(beta3)
        self.weight_decay = trans_to_tensor(weight_decay)
        self.weight_decay_flag = bool(weight_decay)

        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self.compression = compression
        self.init_came_state(beta1)
        self.step = Parameter(initializer(0, [1], mstype.float32), name='afactor_step')
        self.fused_ada_factor = P.FusedAdaFactor(enable_scale_parameter=self.scale_parameter,
                                                 enable_first_moment=self.use_first_moment,
                                                 enable_weight_decay=self.weight_decay_flag)
        if context.get_context("device_target") == "CPU":
            self.use_fused_ada_factor = True
        else:
            self.use_fused_ada_factor = False
        logging.info("Came init completed %s.", self.learning_rate)

    def init_came_state(self, beta1):
        """init came variables"""
        if beta1 > 0:
            self.use_first_moment = True
            self.exp_avg = self._parameters.clone(prefix="exp_avg", init='zeros')
        else:
            self.use_first_moment = False
            self.exp_avg = ParameterTuple([Parameter(Tensor(0.0))] * len(self._parameters))

        self.exp_avg_sq = []
        self.exp_avg_sq_col = []
        self.exp_avg_sq_row = []
        self.exp_avg_insta_col = []
        self.exp_avg_insta_row = []
        for param in self._parameters:
            param_dtype = param.dtype
            param_shape = param.shape
            param_name = param.name
            if len(param_shape) > 1:
                self.exp_avg_sq_row.append(Parameter(initializer(0, shape=param_shape[:-1], dtype=param_dtype),
                                                     name="exp_avg_sq_row_{}".format(param_name)))
                self.exp_avg_sq_col.append(Parameter(initializer(0, shape=param_shape[:-2] + param_shape[-1:],
                                                                 dtype=param_dtype),
                                                     name="exp_avg_sq_col_{}".format(param_name)))
                self.exp_avg_insta_row.append(Parameter(initializer(0, shape=param_shape[:-1], dtype=param_dtype),
                                                        name="exp_avg_insta_row_{}".format(param_name)))
                self.exp_avg_insta_col.append(Parameter(initializer(0, shape=param_shape[:-2] + param_shape[-1:],
                                                                    dtype=param_dtype),
                                                        name="exp_avg_insta_col_{}".format(param_name)))
                self.exp_avg_sq.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                 name="exp_avg_sq_{}".format(param_name)))

            else:
                self.exp_avg_sq_row.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                     name="exp_avg_sq_row_{}".format(param_name)))
                self.exp_avg_sq_col.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                     name="exp_avg_sq_col_{}".format(param_name)))
                self.exp_avg_insta_row.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                        name="exp_avg_insta_row_{}".format(param_name)))
                self.exp_avg_insta_col.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                        name="exp_avg_insta_col_{}".format(param_name)))

                if self.compression:
                    self.exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=mstype.float16),
                                                     name="exp_avg_sq_{}".format(param_name)))
                else:
                    self.exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=param_dtype),
                                                     name="exp_avg_sq_{}".format(param_name)))

        self.exp_avg_sq_row = ParameterTuple(self.exp_avg_sq_row)
        self.exp_avg_sq_col = ParameterTuple(self.exp_avg_sq_col)
        self.exp_avg_insta_row = ParameterTuple(self.exp_avg_insta_row)
        self.exp_avg_insta_col = ParameterTuple(self.exp_avg_insta_col)
        self.exp_avg_sq = ParameterTuple(self.exp_avg_sq)

    @property
    def supports_memory_efficient_fp16(self):
        """
        Support memory efficient for fp16
        """
        return True

    @property
    def supports_flat_params(self):
        """
        Support flatten params
        """
        return False

    @jit
    def construct(self, gradients):
        """construct of came optimizer."""
        gradients = self.flatten_gradients(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        F.assign_add(self.step, 1)
        step = self.step
        beta2t = 1.0 - P.Pow()(step, self.decay_rate)

        if self.use_fused_ada_factor:
            success = self.hyper_map(F.partial(_came_opt, self.fused_ada_factor, self.eps, self.clip_threshold,
                                               self.beta1, beta2t, self.weight_decay, lr),
                                     gradients, self._parameters, self.exp_avg, self.exp_avg_sq_row,
                                     self.exp_avg_sq_col, self.exp_avg_sq)
        else:
            success = self.hyper_map(F.partial(_came_opt, self.eps, self.clip_threshold, self.beta1, beta2t, self.beta3,
                                               self.weight_decay, self.scale_parameter, self.compression,
                                               self.use_first_moment, self.weight_decay_flag, lr),
                                     gradients, self._parameters, self.exp_avg, self.exp_avg_sq_row,
                                     self.exp_avg_sq_col, self.exp_avg_sq, self.exp_avg_insta_row,
                                     self.exp_avg_insta_col)

        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
        if value == 'CPU':
            self.fused_ada_factor.set_device("CPU")
            self.use_fused_ada_factor = True
        else:
            self.use_fused_ada_factor = False
