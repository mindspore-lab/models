import numpy as np
from utils import SUPPORTED_BITS, bit_to_packing, pack, unpack
import mindspore as ms
def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * np.ceil(val1 / val2)) == val1

# axis=1 attn; axis=0 ffn
def quantize(tensor, nbits, group_size=None):
    # nbits, group_size, axis = hqq_config.num_bits, hqq_config.group_size, hqq_config.axis
    # assert nbits in SUPPORTED_BITS, ("nbits=" + str(nbits) + " not supported.")
    # if group_size is not None:
    #     assert is_divisible(tensor.numel(), group_size), (
    #         "group_size should be divisble by the total tensor dimensions. shape: "
    #         + str(tensor.shape)
    #         + ", group_size: "
    #         + str(group_size)
    #     )
    if nbits == 4:
        group_size = 64
    elif nbits == 2:
        group_size = 32

    W = tensor.float()
    shape = W.shape

    # Reshape for grouping
    W =  W.reshape([group_size, -1])
    # Get min/max values
    _min = W.min(axis=0, keepdims=True)[0]
    _max = W.max(axis=0, keepdims=True)[0]

    max_v = 2**nbits - 1
    min_v = 0
    min_max = [min_v, max_v]

    scale = (max_v / (_max - _min)).clamp(max=2e4)
    zero = -_min * scale

    # Quantize
    scale, zero = (ms.ops.deepcopy(scale), ms.ops.deepcopy(zero))  # Necessary for fake quantization backprop
    W_q = ms.ops.round(W * scale + zero).clamp(min_max[0], min_max[1])
    W_q = pack[bit_to_packing[nbits]](W_q)

    # quantized_data = {
    #     'nbits': nbits,
    #     'group_size': group_size,
    #     'shape': shape,
    #     'W_q': W_q,
    #     'scale': 1.0 / scale,
    #     'zero': zero,
    #     "packing": bit_to_packing[nbits],
    #     'compute_dtype': tensor.dtype,
    # }
    quantized_data = {
        'nbits': ms.tensor(nbits),
        'shape': ms.tensor(shape),
        'W_q': W_q,
        'scale': 1.0 / scale,
        'zero': zero,
    }

    # cleanup
    del W, _min, _max
    # torch.cuda.empty_cache()

    return quantized_data

# Main dequantization: bit_unpacking > (W_q - z)*s > reshape
def dequantize(quantized_data):
    W_q, scale, zero = quantized_data['W_q'], quantized_data['scale'], quantized_data['zero']
    # W_q, scale, zero = W_q.data, scale.data, zero.data

    W_q, scale, zero = W_q, scale, zero
    packing = bit_to_packing[int(quantized_data['nbits'].asnumpy())]
    # compute_dtype = quantized_data['compute_dtype'] if (quantized_data['compute_dtype'] is not None) else torch.bfloat16
    compute_dtype = ms.float16
    W_r = unpack[packing](W_q, dtype=compute_dtype)
    # if (quantized_data['group_size'] is not None) and (quantized_data['nbits'] == 3):
    #     W_r = (
    #         W_r[: quantized_data['group_size']]
    #         if (quantized_data['axis'] == 0)
    #         else W_r[:, : quantized_data['group_size']]
    #     )
    # W_r = ((W_r - zero) * scale).reshape(torch.Size(quantized_data['shape'])).to(quantized_data['compute_dtype'])
    W_r = ((W_r - zero) * scale).reshape(tuple(quantized_data['shape'].asnumpy().tolist())).to(compute_dtype)
    return W_r

# def dequantize22(quantized_data):
#     W_q, scale, zero = quantized_data['W_q'], quantized_data['scale'], quantized_data['zero'], 
#     W_q, scale, zero = W_q, scale, zero
#     # compute_dtype = quantized_data['compute_dtype'] if (quantized_data['compute_dtype'] is not None) else torch.float32
#     compute_dtype = torch.bfloat16
#     W_r = unpack[quantized_data['packing']](W_q, dtype=compute_dtype)
#     if (quantized_data['group_size'] is not None) and (quantized_data['nbits'] == 3):
#         W_r = (
#             W_r[: quantized_data['group_size']]
#             if (quantized_data['axis'] == 0)
#             else W_r[:, : quantized_data['group_size']]
#         )
#     W_r = ((W_r - zero) * scale).reshape(quantized_data['shape']).to(quantized_data['compute_dtype'])
#     return W_r

