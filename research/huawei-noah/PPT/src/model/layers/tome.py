# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import mindspore as ms
from mindspore.ops import Reshape, Transpose, BatchMatMul, ArgMaxWithValue, Sort, GatherD
import time
from typing import List, Tuple, Union, Callable
from tqdm import tqdm
from torchprofile import profile_macs
import torch
from mindspore.ops.function import concat, broadcast_to

def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: ms.Tensor,
    r: int,
    class_token: bool = True,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    matmul = BatchMatMul()
    transpose = Transpose()    
    metric = metric / metric.norm(axis=-1, keep_dims=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = matmul(a, transpose(b, (0, 2, 1)))

    if class_token:
        scores[..., 0, :] = -2.0
    if distill_token:
        scores[..., :, 0] = -2.0
        
    node_max, node_idx = ms.Tensor(scores.max(-1)), ms.Tensor(scores.argmax(-1))
    sort = Sort(axis=-1, descending=True)
    edge_idx = sort(node_max)[1][..., None]
    
    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    src_idx = edge_idx[..., :r, :]  # Merged Tokens
    dst_idx = ms.ops.gather_elements(node_idx[..., None], dim=-2, index=src_idx)

    if class_token:
        # Sort to ensure the class token is at the start
        # breakpoint()
        sort = Sort(axis=1)
        idx = sort(unm_idx+0.0)[1]  # ms的sort仅支持float16的输入数据格式，逼我出绝招，目前所有内容都已实现对齐
        unm_idx = ms.ops.gather_elements(unm_idx, dim=1, index=idx)
        

    def merge(x: ms.Tensor, mode="mean") -> ms.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape    # [B, 99, 384]
        
        # breakpoint()
        unm = ms.ops.gather_elements(src, dim=-2, index=broadcast_to(unm_idx,(n, t1 - r, c)))
        src = ms.ops.gather_elements(src, dim=-2, index=broadcast_to(src_idx, (n, r, c)))
        dst = ms.ops.tensor_scatter_elements(dst, broadcast_to(dst_idx, (n, r, c)), src, -2, 'add')
        
        # unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return concat((unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]), axis=1)
        else:
            return concat((unm, dst), axis=1)

    def unmerge(x: ms.Tensor) -> ms.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = ms.ops.Zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: ms.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = split(metric)
    r = a.shape[1]
    scores = a @ b.transpose(-1, -2)

    _, dst_idx = scores.max(dim=-1)
    dst_idx = dst_idx[..., None]

    def merge(x: ms.Tensor, mode="mean") -> ms.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: ms.Tensor) -> ms.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = concat((src, dst), dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: ms.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    B, N, _ = metric.shape
    rand_idx = ms.ops.UniformReal(B, N, 1, device=metric.device).argsort(dim=1)

    a_idx = rand_idx[:, :r, :]
    b_idx = rand_idx[:, r:, :]

    def split(x):
        C = x.shape[-1]
        a = x.gather(dim=1, index=a_idx.expand(B, r, C))
        b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
        return a, b

    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = split(metric)
    scores = a @ b.transpose(-1, -2)

    _, dst_idx = scores.max(dim=-1)
    dst_idx = dst_idx[..., None]

    def merge(x: ms.Tensor, mode="mean") -> ms.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: ms.Tensor) -> ms.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = ms.ops.Zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: ms.Tensor, size: ms.Tensor = None
) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: ms.Tensor, source: ms.Tensor = None
) -> ms.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = ms.ops.Eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def benchmark(
    model: ms.nn.Cell,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = True,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    
    model.set_train(False)
    
    # # FLOPs evaluate
    # input = ms.ops.UniformReal(seed=0)((1, *input_size))
    # macs = profile_macs(model, input)
    # print(f"macs:{macs/1e9}GFLOPs")
    
    # throughput evaluate
    input = ms.ops.UniformReal(seed=0)((batch_size, *input_size))
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
        if i == warm_up:
            total = 0
            start = time.time()

        model(input)
        total += batch_size

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]