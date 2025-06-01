import gc
from mindspore import uint8, int32, Tensor
import numpy as np
import argparse
import GPUtil
import mindspore as ms
import threading
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

KB = ms.ops.bitwise_left_shift(ms.Tensor(1), ms.Tensor(10))  # 1 << 10
MB = ms.ops.bitwise_left_shift(ms.Tensor(1), ms.Tensor(20))  # 1 << 20
GB = ms.ops.bitwise_left_shift(ms.Tensor(1), ms.Tensor(30))  # 1 << 30
T = 1e12

def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if ms.is_tensor(obj) and not obj.is_cuda]

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
    tensors = [obj for obj in objects if ms.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        # print(tensor.shape, tensor.data_ptr())

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem

def memory_cost_qwen(config, memory_budget):
    #初始化
    a=ms.Tensor([1,1])
    b=ms.Tensor([1,1])
    print(a+b)
    if memory_budget == 0:
        total_memory = ms.hal.get_device_properties(0).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        # gpu_info = context.get_gpu_status()
        allocated_memory = ms.hal.memory_stats()["total_allocatd_memory"]     
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024
    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    shared_expert_intermediate_size = config.shared_expert_intermediate_size
    num_experts = config.num_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    max_position_embeddings = config.max_position_embeddings

    # layer
    embed = (vocab_size * hidden_size * 2 * 2)/MB
    attention = (2 * (hidden_size * num_heads * head_dim + 
                        hidden_size * num_key_value_heads * head_dim) 
                        * num_hidden_layers * 2)/MB
    attn_bias = ((num_heads * head_dim + 2 * num_key_value_heads * head_dim) * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    shared_expert = (3 * hidden_size * shared_expert_intermediate_size * 2)/MB
    shared_expert_gate = (hidden_size * num_hidden_layers * num_hidden_layers * 2)/MB

    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB
    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - attn_bias - rotaryEmbedding - norm -
                        shared_expert_gate - expert_gate - kv - hidden)
    available_memory = available_memory - (shared_expert * 24)

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 60 + 3 + 4 + 2 + 2 + 2) * 24
    available_memory = available_memory - meta_data - 1000
    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    
    # all in mem.
    if available_memory > num_hidden_layers * 60 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 60 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 60 * expert_int4)
        fp16_layers = available_memory // (60 * expert - 60 * expert_int4)
        for i in range(num_hidden_layers):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 60
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 60 - (cache_num - 60 * (all_cache_layers))
                else:
                    offload_map[i] = 60
        else:
            cache_deep = (cache_num - 4 * 60) // (num_hidden_layers - 4)
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 60 - cache_deep
        return (offload_map, quan_map)
  
def memory_cost_deepseek(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = ms.hal.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size
    shared_expert_intermediate_size = 2 * config.moe_intermediate_size
    num_experts = config.n_routed_experts

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads
    max_position_embeddings = config.max_position_embeddings

    # layer
    embed = (2 * vocab_size * hidden_size * 2)/MB
    attention = (2 * (hidden_size * num_heads * head_dim + 
                        hidden_size * num_key_value_heads * head_dim) 
                        * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    shared_expert = (3 * hidden_size * shared_expert_intermediate_size * 2)/MB
    dense_expert = (3 * hidden_size * intermediate_size * 2)/MB
    expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention - rotaryEmbedding - norm - expert_gate - kv - hidden)
    available_memory = available_memory - (shared_expert * num_hidden_layers) - dense_expert
    print("available_memory: ", available_memory)

    # remove meta_data, prefetch
    meta_data = 0.3 * (3 * 64 + 3 + 4 + 2 + 2 + 2) * num_hidden_layers
    available_memory = available_memory - meta_data + 200

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
    expert_int4 = expert/4 + 3 * zero_scale

    #
    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    num_hidden_layers = num_hidden_layers - 1
    # all in mem.
    if available_memory > num_hidden_layers * 64 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 64 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 64 * expert_int4)
        fp16_layers = available_memory // (64 * expert - 64 * expert_int4)
        for i in range(num_hidden_layers):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 64
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 64 - (cache_num - 64 * (all_cache_layers))
                else:
                    offload_map[i] = 64
        else:
            cache_deep = (cache_num - 4 * 64) // (num_hidden_layers - 4)
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 64 - cache_deep
        return (offload_map, quan_map)

def memory_cost_jetmoe(config, memory_budget):
    if memory_budget == 0:
        device = config.device
        total_memory = ms.hal.get_device_properties(device).total_memory / (1024**2)
        # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
        gpus = GPUtil.getGPUs()
        target_gpu = next((gpu for gpu in gpus if f"cuda:{gpu.id}" == device), None)
        allocated_memory = target_gpu.memoryUsed
        
        memory_budget = total_memory - allocated_memory
    else:
        memory_budget = memory_budget * 1024

    # Reserve space for 1024
    seq_len = 1024
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    intermediate_size = config.ffn_hidden_size
    num_experts = config.moe_num_experts
    input_size = config.hidden_size
    output_size = config.kv_channels * config.num_key_value_heads

    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    max_position_embeddings = config.max_position_embeddings
    kv_projection_size = config.kv_channels * config.num_key_value_heads

    # layer
    embed = (vocab_size * hidden_size * 2 * 2)/MB
    attention_input_linear = ((num_experts * output_size * input_size) * num_hidden_layers * 2)/MB
    attention_outut_linear = ((num_experts * output_size * input_size) * num_hidden_layers * 2)/MB
    attention_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    kv_proj = (hidden_size * kv_projection_size * 2 * num_hidden_layers * 2)/MB
    bias = (hidden_size * num_hidden_layers * 2)/MB
    rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
    norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

    expert = ((input_size * intermediate_size * 2 + input_size * intermediate_size)* 2)/MB
    expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    moe_bias = (hidden_size * num_hidden_layers * 2)/MB
    expert_gate_cpu = (hidden_size * num_experts * num_hidden_layers * 2)/MB
    
    # during infer.
    kv = (2 * seq_len * num_hidden_layers * hidden_size * 2)/MB
    hidden = (2 * seq_len * hidden_size * 2)/MB

    # Remove frequently accessed layers
    available_memory = (memory_budget - embed - attention_input_linear - attention_outut_linear - attention_gate
                        - kv_proj - bias - rotaryEmbedding - norm - expert_gate - moe_bias - expert_gate_cpu - kv - hidden)

    # remove meta_data, prefetch
    meta_data = 0.3 * (((2 * 8 + 2) + ( 2 * 8 + 3) + 2) * num_hidden_layers + 3)
    available_memory = available_memory - meta_data - 2000
    # print(available_memory)

    if available_memory < 0:
        assert False, f"{available_memory}, memory is not enough for dense."

    zero_scale_input_linear = (2 * (intermediate_size * 2 // 64 * hidden_size) * 4)/MB
    zero_scale_output_linear = (2 * (intermediate_size // 64 * hidden_size)* 4)/MB
    expert_int4 = expert / 4 + zero_scale_input_linear + zero_scale_output_linear

    #
    quan_map = {}
    offload_map = {}
    for i in range(num_hidden_layers):
        quan_map[i] = 0
        offload_map[i] = 0
    
    # all in mem.
    if available_memory > num_hidden_layers * 8 * expert:
        return (offload_map, quan_map)
    # all in mem. with int4
    elif available_memory > (num_hidden_layers * 8 * expert_int4):
        available_memory = available_memory - (num_hidden_layers * 8 * expert_int4)
        fp16_layers = available_memory // (8 * expert - 8 * expert_int4)
        for i in range(num_hidden_layers):
            if i == fp16_layers or i > fp16_layers:
                quan_map[i] = 4
        return (offload_map, quan_map)
    # offload
    else:
        cache_num = available_memory // expert_int4
        all_cache_layers = cache_num // 8
        # weather shallow can be all cached
        if all_cache_layers < 4:
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < all_cache_layers:
                    offload_map[i] = 0
                elif i == all_cache_layers:
                    offload_map[i] = 8 - (cache_num - 8 * (all_cache_layers))
                else:
                    offload_map[i] = 8
        else:
            cache_deep = (cache_num - 4 * 8) // (num_hidden_layers - 4)
            for i in range(num_hidden_layers):
                quan_map[i] = 4
                if i < 4:
                    offload_map[i] = 0
                else:
                    offload_map[i] = 8 - cache_deep
        return (offload_map, quan_map)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def add_task(self, task_id, func, *args, **kwargs):
        event = threading.Event()
        with self.lock:
            self.tasks[task_id] = event
        
        threading.Thread(target=self._run_task, args=(task_id, func, event, *args), kwargs=kwargs).start()

    def _run_task(self, task_id, func, event, *args, **kwargs):
        func(*args, **kwargs)  # 执行任务
        event.set()  # 标记任务完成

    def wait_for_task(self, task_id):
        """等待指定任务完成"""
        with self.lock:
            event = self.tasks.get(task_id)
        if event:
            event.wait()  # 等待任务完成信号
        else:
            print(f"Task {task_id} not found.")

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
        # return (W_q[:_step] << 4) | W_q[_step:]
        return (ms.ops.bitwise_left_shift(W_q[:_step],4)) | W_q[_step:]

    @staticmethod
    def unpack_4bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:  # uint8/2 > uint8
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([2 * _step, W_q.shape[1]], dtype=dtype)

        # tmp[:_step] = ms.ops.bitwise_right_shift(W_q & 0b11110000,4)
        # tmp[_step:] = W_q & 0b00001111
        tmp[:_step] = ms.ops.bitwise_right_shift(ms.ops.bitwise_and(W_q , 0b11110000),4)
        tmp[_step:] = ms.ops.bitwise_and(W_q , 0b00001111)

        return tmp

    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 4)

        return (
            ms.ops.bitwise_left_shift(W_q[:_step] , 6)
            | ms.ops.bitwise_left_shift(W_q[_step : 2 * _step] , 4)
            | ms.ops.bitwise_left_shift(W_q[2 * _step : 3 * _step] , 2)
            | W_q[3 * _step :]
        )

    @staticmethod
    def unpack_2bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([4 * _step, W_q.shape[1]], dtype=dtype)
        ms.ops.bitwise_right_shift(ops.bitwise_and(W_q , 0b11110000),6)
        tmp[0 * _step : 1 * _step] = ms.ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b11000000), 6)
        tmp[1 * _step : 2 * _step] = ms.ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00110000), 4)
        tmp[2 * _step : 3 * _step] = ms.ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00001100), 2)
        tmp[3 * _step : 4 * _step] = ops.bitwise_and(W_q , 0b00000011)

        return tmp

    # 3-bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in: Tensor) -> Tensor:
        W_q = ms.ops.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=int32,
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)

        W_q = (
            ms.ops.bitwise_left_shift(W_q[:_step], 27) |
            ms.ops.bitwise_left_shift(W_q[1 * _step : 2 * _step], 24) |
            ms.ops.bitwise_left_shift(W_q[2 * _step : 3 * _step], 21) |
            ms.ops.bitwise_left_shift(W_q[3 * _step : 4 * _step], 18) |
            ms.ops.bitwise_left_shift(W_q[4 * _step : 5 * _step], 15) |
            ms.ops.bitwise_left_shift(W_q[5 * _step : 6 * _step], 12) |
            ms.ops.bitwise_left_shift(W_q[6 * _step : 7 * _step], 9) |
            ms.ops.bitwise_left_shift(W_q[7 * _step : 8 * _step], 6) |
            ms.ops.bitwise_left_shift(W_q[8 * _step : 9 * _step], 3) |
            W_q[9 * _step : 10 * _step]
        )

        return W_q

    # A bit faster than _cat version
    @staticmethod
    def unpack_3bit_32(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([10 * _step, W_q.shape[1]], dtype=dtype)

        tmp[0 * _step : 1 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00111000000000000000000000000000), 27)
        tmp[1 * _step : 2 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000111000000000000000000000000), 24)
        tmp[2 * _step : 3 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000111000000000000000000000), 21)
        tmp[3 * _step : 4 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000111000000000000000000), 18)
        tmp[4 * _step : 5 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000000111000000000000000), 15)
        tmp[5 * _step : 6 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000000000111000000000000), 12)
        tmp[6 * _step : 7 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000000000000111000000000), 9)
        tmp[7 * _step : 8 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000000000000000111000000), 6)
        tmp[8 * _step : 9 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000000000000000000000000111000), 3)
        tmp[9 * _step : 10 * _step] = ops.bitwise_and(W_q ,0b00000000000000000000000000000111)

        return tmp

    # 1-bit
    ################################################
    @staticmethod
    def pack_1bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 8)

        return (
                ms.ops.bitwise_left_shift(W_q[:_step], 7) |
                ms.ops.bitwise_left_shift(W_q[1 * _step : 2 * _step], 6) |
                ms.ops.bitwise_left_shift(W_q[2 * _step : 3 * _step], 5) |
                ms.ops.bitwise_left_shift(W_q[3 * _step : 4 * _step], 4) |
                ms.ops.bitwise_left_shift(W_q[4 * _step : 5 * _step], 3) |
                ms.ops.bitwise_left_shift(W_q[5 * _step : 6 * _step], 2) |
                ms.ops.bitwise_left_shift(W_q[6 * _step : 7 * _step], 1) |
                W_q[7 * _step : 8 * _step]
            )

    @staticmethod
    def unpack_1bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = ms.numpy.empty([8 * _step, W_q.shape[1]], dtype=dtype)

        tmp[0 * _step : 1 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b10000000), 7)
        tmp[1 * _step : 2 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b01000000), 6)
        tmp[2 * _step : 3 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00100000), 5)
        tmp[3 * _step : 4 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00010000), 4)
        tmp[4 * _step : 5 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00001000), 3)
        tmp[5 * _step : 6 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000100), 2)
        tmp[6 * _step : 7 * _step] = ops.bitwise_right_shift(ops.bitwise_and(W_q, 0b00000010), 1)
        tmp[7 * _step : 8 * _step] = ops.bitwise_and(W_q ,0b00000001)

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