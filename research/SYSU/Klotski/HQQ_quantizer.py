import dataclasses

import torch
import numpy as np
from pytorch_backend import (TorchTensor, DeviceType)
from utils import np_dtype_to_torch_dtype
from utils import SUPPORTED_BITS, bit_to_packing, pack, unpack
import copy
@dataclasses.dataclass
class HQQConfig:
    def __init__(self, num_bits=None, group_size=None, axis=None, shape=None, packing=None, compute_dtype=None, quantize_shape=None):
        self.num_bits = num_bits
        self.group_size = group_size
        self.shape = shape
        self.packing = packing
        self.compute_dtype = compute_dtype
        self.quantize_shape = quantize_shape
        self.axis = axis

def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * np.ceil(val1 / val2)) == val1

class TorchHQQDevice:
    def __init__(self, base_device):
        self.name = "compressed"
        self.device_type = DeviceType.QUANTIZE
        self.base_device = base_device

    def allocate(self, shape, dtype, hqq_config, pin_memory=None, name=None):
        if hqq_config.axis == 1: # attn
            data_shape = ((shape[1]//hqq_config.group_size)*shape[0]//(8//hqq_config.num_bits), hqq_config.group_size)
            scale_zero_shape = ((shape[1]//hqq_config.group_size)*shape[0], 1)
        else: # expert
            data_shape = ((hqq_config.group_size//(8//hqq_config.num_bits)), (shape[1]//hqq_config.group_size)*shape[0])
            scale_zero_shape = (1, (shape[1]//hqq_config.group_size)*shape[0])
        data = self.base_device.allocate(data_shape, np.uint8, pin_memory=pin_memory)
        scale = self.base_device.allocate(scale_zero_shape, np.float32, pin_memory=pin_memory)
        zero = self.base_device.allocate(scale_zero_shape, np.float32, pin_memory=pin_memory)
        hqq_config_copy = copy.deepcopy(hqq_config)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype], (data, scale, zero, hqq_config_copy), self, name=name)

    def quantize(self, tensor, hqq_config):
        nbits, group_size, axis = hqq_config.num_bits, hqq_config.group_size, hqq_config.axis
        assert nbits in SUPPORTED_BITS, ("nbits=" + str(nbits) + " not supported.")
        assert axis in [0, 1], "axis should be either 0 or 1"
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )

        W = tensor.float()
        shape = W.shape

        # Reshape for grouping
        W = (
                W.reshape([-1, group_size])
                if (axis == 1)
                else W.reshape([group_size, -1])
            )

        # Get min/max values
        _min = W.min(axis=axis, keepdim=True)[0]
        _max = W.max(axis=axis, keepdim=True)[0]

        max_v = 2**nbits - 1
        min_v = 0
        min_max = [min_v, max_v]

        scale = (max_v / (_max - _min)).clamp(max=2e4)
        zero = -_min * scale

        # Quantize
        scale, zero = (scale.clone(), zero.clone(),)  # Necessary for fake quantization backprop
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

        W_q = pack[bit_to_packing[nbits]](W_q)

        # Store meta-data (we invert the scale for dequantization)
        hqq_config.shape = shape
        hqq_config.packing = bit_to_packing[nbits]
        hqq_config.compute_dtype = tensor.dtype
        hqq_config.quantize_shape = W_q.shape

        # cleanup
        del W, _min, _max
        torch.cuda.empty_cache()

        W_q = TorchTensor.create_from_torch(W_q, self.base_device)
        scale = TorchTensor.create_from_torch(1.0 / scale, self.base_device)
        zero = TorchTensor.create_from_torch(zero, self.base_device)

        return TorchTensor(shape, tensor.dtype, (W_q, scale, zero, hqq_config), self)

    # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
    def dequantize(self, tensor):
        W_q, scale, zero, hqq_config = tensor.data
        W_q, scale, zero = W_q.data, scale.data, zero.data
        compute_dtype = hqq_config.compute_dtype if (hqq_config.compute_dtype is not None) else torch.float32
        W_r = unpack[hqq_config.packing](W_q, dtype=compute_dtype)
        if (hqq_config.group_size is not None) and (hqq_config.num_bits == 3):
            W_r = (
                W_r[: hqq_config.group_size]
                if (hqq_config.axis == 0)
                else W_r[:, : hqq_config.group_size]
            )
        W_r = ((W_r - zero) * scale).reshape(hqq_config.shape).to(hqq_config.compute_dtype)
        return W_r

