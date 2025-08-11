# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils"""
import re
import numpy as np

import mindspore as ms
from mindspore.ops import operations as P
from mindspore import nn, Tensor
from mindformers.tools.logger import logger

__all__ = ["get_attn_mask_func"]

MAX_INT32 = 2147483647


class AttnMaskFill(nn.Cell):
    """Applies a mask to attention scores by filling masked positions with a specified value."""

    def __init__(self, config):
        super(AttnMaskFill, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.add = P.Add()
        self.shard(config)

    def construct(self, attention_scores: Tensor, attention_mask, fill_value=-10000.0):
        """ Construct function of AttnMaskFill. """
        ori_dtype = attention_scores.dtype
        if attention_mask.ndim == 2:
            bs, _, seq_length0, seq_length1 = self.shape(attention_scores)
            attention_mask = self.reshape(attention_mask, (1, 1, seq_length0, seq_length1))
            attention_mask = self.tile(attention_mask, (bs, 1, 1, 1))

        # socres * lower_triangle + attention_mask * -10000.0
        ones = Tensor([1.0], dtype=attention_mask.dtype)
        lower_triangle = self.sub(ones, attention_mask)
        attention_scores = self.mul(attention_scores, lower_triangle)
        attention_scores = self.add(attention_scores, self.mul2(attention_mask, fill_value))
        attention_scores = self.cast(attention_scores, ori_dtype)
        return attention_scores

    def shard(self, parallel_config):
        """sharding parameters"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        tp = 1 if parallel_config is None else parallel_config.tensor_parallel

        self.tile.shard(((dp, 1, 1, 1),))
        self.sub.shard(((1,), (dp, 1, 1, 1)))
        self.mul.shard(((dp, tp, 1, 1), (dp, 1, 1, 1)))
        self.mul2.shard(((dp, 1, 1, 1), ()))
        self.add.shard(((dp, tp, 1, 1), (dp, 1, 1, 1)))


class AttnMaskAdd(nn.Cell):
    """Adds a mask to attention scores by adding the mask values to the attention scores."""

    def __init__(self, config):
        super(AttnMaskAdd, self).__init__()
        self.add = P.Add()
        self.cast = P.Cast()
        self.shard(config)

    def construct(self, attention_scores: Tensor, attention_mask):
        """ Construct function of AttnMaskAdd. """
        attention_scores = self.add(attention_scores, self.cast(attention_mask, attention_scores.dtype))

        return attention_scores

    def shard(self, parallel_config):
        """sharding parameters"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        tp = 1 if parallel_config is None else parallel_config.tensor_parallel
        cp = 1 if parallel_config is None else parallel_config.context_parallel

        self.add.shard(((dp, tp, cp, 1), (cp, 1)))


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": AttnMaskFill,
    "attn_mask_add": AttnMaskAdd,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill', 'attn_mask_add'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]


class LayerSetting:
    r"""
    Class for setting offset, pipeline stage and select recompute for each transformer layer.
    Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            num_layers(int) - The total layers in the model.
            offset(Union[int, List[int], List[List[int]]]) - The layer offset for each (mini) stage.
            parallel_config(dict) - Parallel Config.
            pp_interleave_num(int) - The number of pp-interleave. When it is larger than 1,
                each stage will be divided into mini stages by pp_interleave_num.
    """

    def __init__(self, num_layers, offset, parallel_config, pp_interleave_num=1):
        default_patterns = [r'feed_forward\.mul', r'feed_forward\.w1\.activation\.silu']
        default_comm_patterns = [r'.*\.norm']
        try:
            use_pp_interleave = ms.get_auto_parallel_context("pipeline_interleave")
        except ValueError:
            logger.error("Current MindSpore version do not pipeline interleave. pp_interleave_num is set to 1.")
            use_pp_interleave = False
        self.num_layers = num_layers
        self.pp = parallel_config.pipeline_stage
        self.recompute = parallel_config.recompute
        self.gradient_aggregation_group = parallel_config.gradient_aggregation_group
        self.pp_interleave_num = pp_interleave_num if use_pp_interleave else 1
        self.offset = np.array(offset, np.int64)
        self._check_inputs()
        self.offset = np.broadcast_to(self.offset, (self.pp_interleave_num, self.pp))

        avg_layer = (self.num_layers + 1) // (self.pp * self.pp_interleave_num)
        self.layer_list = np.ones((self.pp_interleave_num, self.pp), np.int64) * avg_layer + self.offset
        interleave_sum = np.insert(np.cumsum(np.sum(self.layer_list, axis=1))[:-1], 0, 0)
        self.layer_accu = np.cumsum(self.layer_list, axis=1) + interleave_sum.reshape(-1, 1)
        self.pp_ids = [np.searchsorted(self.layer_accu.reshape(-1), i + 1) % self.pp for i in range(self.num_layers)]
        self.interleave_ids = [np.where(i < self.layer_accu)[0][0] for i in range(self.num_layers)]
        logger.info("num_layers per stage: %s", self.layer_list.tolist())
        logger.info("Accumulated num_layers per stage: %s", self.layer_accu.tolist())
        logger.info("Pipeline id list: %s", self.pp_ids)
        logger.info("Interleave id list: %s", self.interleave_ids)
        pre_pad = np.array([[0] + [self.layer_accu[i, -1] for i in range(len(self.layer_accu) - 1)]])
        self.layer_accu_mod = np.concatenate((pre_pad.T, self.layer_accu), axis=-1)

        if not isinstance(self.recompute, bool):
            self.layer_recompute = self._format_recompute_list(self.recompute.recompute)
            self.select_recompute = self._format_recompute_dict(
                self.recompute.select_recompute, default_patterns)
            self.select_comm_recompute = self._format_recompute_dict(
                self.recompute.select_comm_recompute, default_comm_patterns)
            logger.info("Formative layer_recompute: %s", self.layer_recompute)
            logger.info("Formative select_recompute: %s", self.select_recompute)
            logger.info("Formative select_comm_recompute: %s", self.select_comm_recompute)

    def set(self, layer, layer_id):
        """Set pipeline stage and recompute for each layer with a layer_id."""
        pp_id = int(self.pp_ids[layer_id])
        layer.pipeline_stage = pp_id
        dis = max(int((self.num_layers + 1) / self.gradient_aggregation_group), 1)
        if self.pp > 1:
            layer.set_comm_fusion(2)
        else:
            layer.set_comm_fusion(int((layer_id + self.offset[0, 0]) / dis) + 1)

        if isinstance(self.recompute, bool):
            if self.recompute:
                layer.recompute()
            return
        if self.recompute.recompute:
            if isinstance(self.recompute.recompute, bool):
                if self.recompute.recompute:
                    layer.recompute(recompute_slice_activation=self.recompute.recompute_slice_activation)
            else:
                if self._check_layer_rule(layer_id):
                    layer.recompute()
                    logger.info("Set full recompute at layer %s", layer_id)
                else:
                    self._set_select_recompute(layer, layer_id, False)
                    self._set_select_recompute(layer, layer_id, True)
        else:
            self._set_select_recompute(layer, layer_id, False)
            self._set_select_recompute(layer, layer_id, True)

    def _alloc_recompute_layer(self, select_recompute):
        """Average allocate recompute layer among different interleave."""
        res = np.zeros_like(self.layer_list)
        layer_by_stage = np.sum(self.layer_list, axis=0)
        for j, num in enumerate(select_recompute):
            num = min(num, layer_by_stage[j])
            while num:
                for i in range(self.pp_interleave_num):
                    if not num:
                        break
                    if res[i, j] < self.layer_list[i, j]:
                        res[i, j] += 1
                        num -= 1
        return res

    def _format_recompute_list(self, select_recompute):
        """Format recompute inputs into a list."""
        if isinstance(select_recompute, bool):
            if select_recompute:
                return self.layer_list.tolist()
            return np.zeros_like(self.layer_list).tolist()
        if isinstance(select_recompute, (list, tuple)):
            if all(isinstance(item, (int, bool)) for item in select_recompute):
                for i, item in enumerate(select_recompute):
                    if isinstance(item, bool):
                        select_recompute[i] = MAX_INT32 if item else 0
                return self._alloc_recompute_layer(select_recompute).tolist()
            if all(isinstance(item, str) for item in select_recompute):
                return self.layer_list.tolist()
            for i, sub in enumerate(select_recompute):
                if isinstance(sub, bool):
                    select_recompute[i] = self.layer_list[i] if sub else np.zeros_like(self.layer_list[i])
                elif isinstance(sub, (list, tuple)):
                    for j, item in enumerate(sub):
                        if isinstance(item, bool):
                            select_recompute[i][j] = self.layer_list[i][j] if item else 0
                        elif isinstance(item, int):
                            select_recompute[i][j] = min(select_recompute[i][j], self.layer_list[i, j])
                        else:
                            raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
                else:
                    raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")
            return select_recompute
        raise ValueError(f"Illegal input list for select_recompute: {select_recompute}")

    def _format_recompute_dict(self, select_recompute, default_patterns):
        """Format select_recompute inputs into a dict"""
        dic = {}
        if isinstance(select_recompute, (list, tuple)) and all(isinstance(item, str) for item in select_recompute):
            parttern = select_recompute
        elif isinstance(select_recompute, dict):
            parttern = select_recompute.keys()
        else:
            parttern = default_patterns
        for p in parttern:
            recompute_value = select_recompute[p] if isinstance(select_recompute, dict) else select_recompute
            dic[p] = self._format_recompute_list(recompute_value)
        return dic

    def _check_layer_rule(self, layer_id):
        """Check whether a layer should be set full layer recompute."""
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < self.layer_recompute[v_id][pp_id]:
            return True
        return False

    def _set_select_recompute(self, layer, layer_id, add_prim_attr=False):
        """Set select recompute for a layer."""
        select_recompute = self.select_comm_recompute if add_prim_attr else self.select_recompute
        pp_id = int(self.pp_ids[layer_id])
        v_id = int(self.interleave_ids[layer_id])
        log_ops = []
        for pattern, layers_dict in select_recompute.items():
            if 0 <= layer_id - self.layer_accu_mod[v_id, pp_id] < layers_dict[v_id][pp_id]:
                log = LayerSetting.set_pattern_recompute(layer, pattern.split(r'\.'), add_prim_attr)
                if log:
                    log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info("Set select %s recompute at layer %s: %s", comm, layer_id, log_ops_str)

    def _check_inputs(self):
        """Check the inputs of offset."""
        if self.offset.ndim >= 1 and self.offset.shape[-1] != self.pp:
            raise ValueError(f"offset.shape[-1] should equal to `pp` ({self.pp}), "
                             f"but got ({self.offset.shape[-1]}). `offset`: {self.offset}")
        if self.offset.ndim >= 2 and self.offset.shape[-2] != self.pp_interleave_num:
            raise ValueError(f"offset.shape[-2] should equal to `pp_interleave_num` ({self.pp_interleave_num}), "
                             f"but got ({self.offset.shape[-2]}). `offset`: {self.offset}")
        if self.offset.sum() != self.num_layers % (self.pp * self.pp_interleave_num):
            r = self.num_layers % (self.pp * self.pp_interleave_num)
            raise ValueError(f"The sum of `offset` ({self.offset.sum()}) should equal to remainder of `num_layers` "
                             f"({self.num_layers}) % (pp ({self.pp}) * pp_interleave_num ({self.pp_interleave_num})) "
                             f"= {r}")

    @staticmethod
    def set_pattern_recompute(layer, p_list, add_prim_attr=False, info=''):
        """Set an operator recompute for a given key-value pair in select_recompute dict."""
        log_list = []
        log = ''
        if p_list:
            p = p_list.pop(0)
        else:
            return info
        if p_list:
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    log = LayerSetting.set_pattern_recompute(cell, p_list, add_prim_attr, info + f'.{name}')
                    if log:
                        log_list.append(log[1:])
        else:
            for attr in dir(layer):
                if re.fullmatch(p, attr):
                    operator = getattr(layer, attr)
                    if add_prim_attr:
                        operator.add_prim_attr("recompute_comm_op", True)
                        log = f"{info}.{attr}"
                    elif hasattr(operator, 'recompute'):
                        operator.recompute()
                        log = f"{info}.{attr}"
            # pylint: disable=W0212
            for name, cell in layer._cells.items():
                if re.fullmatch(p, name):
                    if not add_prim_attr:
                        cell.recompute()
                        log = f"{info}.{name}"
        p_list.insert(0, p)
        if log_list:
            return " " + ", ".join(log_list)
        return log

    def __call__(self, layer, layer_id):
        self.set(layer, layer_id)
