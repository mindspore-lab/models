import argparse
import dataclasses
from attr import define, field
from attr.setters import frozen
import functools
import gc
import os
from typing import Union, Optional, Any, List
import torch
from torch import uint8, int32, Tensor
import numpy as np
import mindspore as ms
from mindspore import ops

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12


@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int

    do_sample: bool
    temperature: float
    stop: Optional[int]


@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    Ascend: Any = None

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        from Klotski.pytorch_backend import TorchDevice, TorchDisk
        gpu = TorchDevice("cuda")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        Ascend = TorchDevice("Ascend")
        return cls(gpu=gpu, cpu=cpu, disk=disk,Ascend=Ascend)

    def close_copy_threads(self):
        self.disk.close_copy_threads()


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark results."""
    prefill_latency: float
    prefill_throughput: float
    decode_latency: float
    decode_throughput: float
    total_latency: float
    total_throughput: float

# float32本来对应bf16
np_dtype_to_torch_dtype = {
    np.float16: ms.dtype.float16, np.float32: ms.dtype.float32, np.uint8: ms.dtype.uint8,
    np.int8: ms.dtype.int8, np.int32: ms.dtype.int32, np.int64: ms.dtype.int64,
    bool: ms.dtype.bool_,ms.float16:ms.float16,ms.float32: ms.float32
}

torch_dtype_to_np_dtype = {
    ms.dtype.float16: np.float16, ms.dtype.float32: np.float32,
    ms.dtype.uint8: np.uint8, ms.dtype.int8: np.int8, ms.dtype.int32: np.int32,
    ms.dtype.int64: np.int64, ms.dtype.bool_: bool,
    ms.dtype.bfloat16: np.float32,
}

torch_dtype_to_num_bytes = {
    ms.dtype.float16: 2, ms.dtype.float32: 4,
    ms.dtype.int8: 1, ms.dtype.uint8: 1, ms.dtype.int32: 4, ms.dtype.int64: 8,
    ms.dtype.Bool: 1,
}


def piecewise_linear_func(xs, ys):
    """Return a function created by linear inerpolation."""
    indices = np.argsort(xs)
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # pad left and right
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    return functools.partial(piecewise_linear_func_ret_func, xs, ys)


def piecewise_linear_func_ret_func(xs, ys, x):
    assert x >= xs[0] and x <= xs[-1]
    return np.interp(x, xs, ys)


def sample_from_range(n, k):
    assert n >= 1

    if k == -1:
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1: return [1]
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))


def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if ops.is_tensor(obj)]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.untyped_storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.untyped_storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


def torch_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if ops.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


class ValueHolder:
    def __init__(self):
        self.val = None

    def store(self, val):
        if self.val is None:
            self.val = val
        else:
            for k, v in val.items():
                self.val[k] = v

    def pop(self):
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        self.val = None

def array_1d(a, cls):
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[S, B, H]
        indices: Tensor[K, B]
    Returns:
        Tensor[K, B, H]
    """
    S, B, H = vectors.shape
    K, B2 = indices.shape
    assert B == B2
    indices = indices.reshape(K, B, 1).expand(K, B, H)
    out = vectors.gather(dim=0, index=indices)
    return out


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):
    log_str = (f"model size: {model_size/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str


def read_benchmark_log(filename):
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]
        throughput = b[b.index(":") + 1:b.index(" to")]
        return float(latency), float(throughput)

    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )


# Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
    # 8-bit
    ################################################
    @staticmethod
    def pack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q.to(uint8)

    @staticmethod
    def unpack_8bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        return W_q.to(dtype)

    # 4-bit
    ################################################
    @staticmethod
    def pack_4bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/2
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 2)

        return (W_q[:_step] << 4) | W_q[_step:]

    @staticmethod
    def unpack_4bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:  # uint8/2 > uint8
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([2 * _step, W_q.shape[1]], dtype=dtype)

        tmp[:_step] = (W_q & 0b11110000) >> 4
        tmp[_step:] = W_q & 0b00001111

        return tmp

    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 4)

        return (
            W_q[:_step] << 6
            | W_q[_step : 2 * _step] << 4
            | W_q[2 * _step : 3 * _step] << 2
            | W_q[3 * _step :]
        )

    @staticmethod
    def unpack_2bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([4 * _step, W_q.shape[1]], dtype=dtype)

        tmp[0 * _step : 1 * _step] = (W_q & 0b11000000) >> 6
        tmp[1 * _step : 2 * _step] = (W_q & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (W_q & 0b00001100) >> 2
        tmp[3 * _step : 4 * _step] = W_q & 0b00000011

        return tmp

    # 3-bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in: Tensor) -> Tensor:
        W_q = ops.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
            dtype=int32,
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)

        W_q = (
            (W_q[:_step] << 27)
            | (W_q[1 * _step : 2 * _step] << 24)
            | (W_q[2 * _step : 3 * _step] << 21)
            | (W_q[3 * _step : 4 * _step] << 18)
            | (W_q[4 * _step : 5 * _step] << 15)
            | (W_q[5 * _step : 6 * _step] << 12)
            | (W_q[6 * _step : 7 * _step] << 9)
            | (W_q[7 * _step : 8 * _step] << 6)
            | (W_q[8 * _step : 9 * _step] << 3)
            | (W_q[9 * _step : 10 * _step])
        )

        return W_q

    # A bit faster than _cat version
    @staticmethod
    def unpack_3bit_32(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([10 * _step, W_q.shape[1]], dtype=dtype)

        tmp[0 * _step : 1 * _step] = (W_q & 0b00111000000000000000000000000000) >> 27
        tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
        tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
        tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
        tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
        tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
        tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
        tmp[9 * _step : 10 * _step] = W_q & 0b00000000000000000000000000000111

        return tmp

    # 1-bit
    ################################################
    @staticmethod
    def pack_1bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 8)

        return (
            W_q[:_step] << 7
            | W_q[1 * _step : 2 * _step] << 6
            | W_q[2 * _step : 3 * _step] << 5
            | W_q[3 * _step : 4 * _step] << 4
            | W_q[4 * _step : 5 * _step] << 3
            | W_q[5 * _step : 6 * _step] << 2
            | W_q[6 * _step : 7 * _step] << 1
            | W_q[7 * _step : 8 * _step]
        )

    @staticmethod
    def unpack_1bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([8 * _step, W_q.shape[1]], dtype=dtype)

        tmp[0 * _step : 1 * _step] = (W_q & 0b10000000) >> 7
        tmp[1 * _step : 2 * _step] = (W_q & 0b01000000) >> 6
        tmp[2 * _step : 3 * _step] = (W_q & 0b00100000) >> 5
        tmp[3 * _step : 4 * _step] = (W_q & 0b00010000) >> 4
        tmp[4 * _step : 5 * _step] = (W_q & 0b00001000) >> 3
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000100) >> 2
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000010) >> 1
        tmp[7 * _step : 8 * _step] = W_q & 0b00000001

        return tmp


SUPPORTED_BITS = [8, 4, 3, 2, 1]

bit_to_packing = {
        8: "8bit_u8",
        4: "4bit_u8",
        3: "3bit_32",
        2: "2bit_u8",
        1: "1bit_u8",
    }

pack = {
    "8bit_u8": BitPack.pack_8bit_u8,
    "4bit_u8": BitPack.pack_4bit_u8,
    "3bit_32": BitPack.pack_3bit_32,
    "2bit_u8": BitPack.pack_2bit_u8,
    "1bit_u8": BitPack.pack_1bit_u8,
}

unpack = {
    "8bit_u8": BitPack.unpack_8bit_u8,
    "4bit_u8": BitPack.unpack_4bit_u8,
    "3bit_32": BitPack.unpack_3bit_32,
    "2bit_u8": BitPack.unpack_2bit_u8,
    "1bit_u8": BitPack.unpack_1bit_u8,
}

unpack_view_dtype = {
    "8bit_u8": uint8,
    "4bit_u8": uint8,
    "3bit_32": int32,
    "2bit_u8": uint8,
    "1bit_u8": uint8,
}