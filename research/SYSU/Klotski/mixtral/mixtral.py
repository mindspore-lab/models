from typing import List, Optional, Union
import dataclasses
import os
import json
import numpy as np
from transformers import AutoTokenizer
import copy
from tqdm import tqdm
import datasets
import random
import time
from HQQ_quantizer import HQQConfig
from mixtral.configuration_mixtral import MixtralConfig, get_mixtral_config, download_mixtral_weights
from pytorch_backend import (TorchDevice, TorchDisk, DeviceType, general_copy, fix_recursive_import)
from time_record import timers
from utils import (Task, ExecutionEnv, GB, ValueHolder, array_1d, array_2d, array_3d, write_benchmark_log)
import mindspore as ms
from mindspore import Tensor, Parameter, mint
fix_recursive_import()
from mindnlp.core import  get_default_dtype

countttMoe=0

def init_weight_list(weight_specs, policy, offload_dst, env, torch_dtype):
    dev_choices = [env.gpu, env.cpu, env.disk]
    ret = []
    for i in range(len(weight_specs)):
        home = dev_choices[offload_dst]
            
        shape, dtype, _ = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            quantize = False
        else:
            pin_memory = policy.pin_weight
            quantize = policy.HQQquantize_weight

        if quantize and (len(weight_specs) == 5 or len(weight_specs) == 3):
            if len(weight_specs) == 5 and i < 4: # quant attn w
                weight = home.quantized_device.allocate(shape, dtype, policy.HQQ_attn_config, pin_memory=pin_memory)
                weight.load_from_np_file(weight_specs[i][2], torch_dtype=torch_dtype)
            elif len(weight_specs) == 3:
                weight = home.quantized_device.allocate(shape, dtype, policy.HQQ_expert_config, pin_memory=pin_memory)
                weight.load_from_np_file(weight_specs[i][2], torch_dtype=torch_dtype)
            else: # not quant ln in attn
                weight = home.allocate(shape, dtype, pin_memory=pin_memory, isMixtralWeight=True)
                weight.load_from_np_file(weight_specs[i][2], torch_dtype=torch_dtype)
        else: # don't quant
            weight = home.allocate(shape, dtype, pin_memory=pin_memory, isMixtralWeight=True)
            weight.load_from_np_file(weight_specs[i][2], torch_dtype=torch_dtype)

        ret.append(weight)
    return ret

@dataclasses.dataclass(frozen=False)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int
    
    offload_expert_weight: int
    offload_dense_weight: int
    offload_kvcache: int
    # offload_hidden: int

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # use sinkToken of attention weights
    attn_sinkToken: bool

    # Quantize weights with HQQ
    HQQquantize_weight: bool
    HQQ_attn_config: HQQConfig
    HQQ_expert_config: HQQConfig


class MixtralInputEmbedding:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.Ascend
        self.weight_load_dst = self.compute
        self.offload_dst = self.policy.offload_dense_weight
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.offload_dst, self.env, self.config.torch_dtype)
        
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf):
        w_token = weight_home.val[0]
        dst = self.weight_load_dst
        # w_token.data = w_token.data.move_to(dst.name)
        # weight_read_buf.store(w_token)
        weight_read_buf.store(w_token.smart_copy(dst, isMixtralWeight=True))

    def forward(self, hidden, weight_read_buf, cur_gate_buf, attention_mask, k):
        # Compute input embedding
        donate = [False] * 3
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)
        
        # mask, donate[1] = attention_mask.val.data.move_to(self.compute.name)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            w_token, donate[2] = weight_read_buf.pop()
        else:
            w_token, _ = weight_read_buf.val

        h = self.compute.mixtral_input_embed(h, mask, w_token, self.config.pad_token_id, donate)
        hidden.val = h


class MixtralOutputEmbedding:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.Ascend
        self.weight_load_dst = self.compute
        self.offload_dst = self.policy.offload_dense_weight
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_final_ln
            ((h,), dtype, path + "norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight")
        ]
        weights = init_weight_list(weight_specs, self.policy, self.offload_dst, self.env, self.config.torch_dtype)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf):
        w_ln, w_token = weight_home.val
        dst1 = self.weight_load_dst
        dst2 = self.compute
        # weight_read_buf.store((w_ln.data.move_to(dst2), w_token.data.move_to(dst1)))
        weight_read_buf.store((w_ln.smart_copy(dst2, isMixtralWeight=True), w_token.smart_copy(dst1, isMixtralWeight=True)))

    def forward(self, hidden, weight_read_buf, cur_gate_buf, attention_mask, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.mixtral_output_embed(h, w_ln, w_token, donate, self.config.rms_norm_eps, self.task.do_sample, self.task.temperature)
        hidden.val = h
        # print("output_emb")
        # print(h.data)

class MixtralRotaryEmbedding():
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ms.ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,  dtype=ms.float16
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ms.ops.arange(self.max_seq_len_cached, dtype=ms.int64).astype(self.inv_freq.dtype)

        freqs = ms.ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ms.ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
    def __call__(self, x, seq_len=None):
        return self.forward(x, seq_len)
    
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        ) 

class MixtralAttention:
    def __init__(self, config, env, policy, layer_idx):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = self.env.Ascend
        self.weight_load_dst = (self.compute.quantized_device if policy.HQQquantize_weight
                                else self.compute)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sliding_window = 4096
        self.sink_num = 4 if policy.attn_sinkToken else 0

        self.task = None
        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.offload_dst = self.policy.offload_dense_weight

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        dtype = self.config.dtype
        path = os.path.join(path, f"layers.{self.layer_idx}.")
        weight_specs = [
            # w_q
            ((self.hidden_size, self.num_heads * self.head_dim), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((self.num_key_value_heads * self.head_dim, self.hidden_size), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((self.num_key_value_heads * self.head_dim, self.hidden_size), dtype, path + "self_attn.v_proj.weight"),
            # w_o
            ((self.num_heads * self.head_dim, self.hidden_size), dtype, path + "self_attn.o_proj.weight"),
            # w_input_ln
            ((self.hidden_size,), dtype, path + "input_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.offload_dst, self.env, self.config.torch_dtype)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf):
        w_q, w_k, w_v, w_o, w_input_ln = weight_home.val
        dst1 = self.weight_load_dst
        dst2 = self.compute
        # weight_read_buf.store((
        #     w_q.data.move_to(dst1), w_k.data.move_to(dst1),
        #     w_v.data.move_to(dst1), w_o.data.move_to(dst1),
        #     w_input_ln.data.move_to(dst2)
        # ))
        weight_read_buf.store((
            w_q.smart_copy(dst1, isMixtralWeight=True), w_k.smart_copy(dst1, isMixtralWeight=True),
            w_v.smart_copy(dst1, isMixtralWeight=True), w_o.smart_copy(dst1, isMixtralWeight=True),
            w_input_ln.smart_copy(dst2, isMixtralWeight=True),
        ))
        if w_q.device.device_type == DeviceType.QUANTIZE:
            ((w_q, d_q), (w_k, d_k), (w_v, d_v), (w_o, d_o), (w_input_ln, d_ln)) = weight_read_buf.val
            w_q = w_q.device.dequantize(w_q)
            w_k = w_k.device.dequantize(w_k)
            w_v = w_v.device.dequantize(w_v)
            w_o = w_o.device.dequantize(w_o)
            weight_read_buf.val = ((w_q, d_q), (w_k, d_k), (w_v, d_v), (w_o, d_o), (w_input_ln, d_ln))

    def init_cache_one_gpu_batch(self, cache_home):
        dev_choices = [self.env.Ascend, self.env.cpu, self.env.disk]
        if isinstance(self.policy.offload_kvcache, int):
            offload_dst = self.policy.offload_kvcache
        else:
            offload_dst = self.policy.offload_kvcache[self.layer_idx]
        device = dev_choices[offload_dst]
        
        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
        a,b=cache_home.val


    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val
        dst = self.compute

        # shape: (s, b * n_head, head_dim)
        indices = (slice(0, self.task.prompt_len + i), slice(0, k_home.shape[1]))
        cache_read_buf.store((
            k_home.smart_copy(dst, indices),
            v_home.smart_copy(dst, indices),
        ))

    def store_cache(self, cache_home, cache_write_buf, i):
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, position_ids, i, k):
        donate = [False] * 10

        h, donate[0] = hidden.val, True
        
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), (w_o, donate[5]),(w_input_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _), (w_v, _), (w_o, _), (w_input_ln, _)) = weight_read_buf.val
        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, past_key_value = self.compute.mixtral_mha(h, mask, w_q, w_k, w_v, w_o, w_input_ln,
                                                         self.num_heads, self.num_key_value_heads, donate,
                                                         self.rotary_emb, self.sliding_window, self.sink_num, position_ids)
            cache_write_buf.store(past_key_value)
        else:  # decoding
            
            (k_cache, donate[7]), (v_cache, donate[8]) = cache_read_buf.pop()

            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, past_key_value = self.compute.mixtral_mha_gen(h, mask, w_q, w_k, w_v, w_o, w_input_ln,
                                                             self.num_heads, self.num_key_value_heads, donate,
                                                             self.rotary_emb, self.sliding_window, self.sink_num, 
                                                             position_ids, k_cache, v_cache)
                                                    
            cache_write_buf.store(past_key_value)

        hidden.val = h

class MixtralTop2Gate:
    def __init__(self, config, env, policy, layer_idx):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = self.env.Ascend
        self.weight_load_dst = self.compute
        self.offload_dst = self.policy.offload_dense_weight
        self.num_local_experts = config.num_local_experts
    
    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.hidden_size, self.num_local_experts, self.config.dtype)
        path = os.path.join(path, f"layers.{self.layer_idx}.")
        weight_specs = [
            # w_ffn_ln
            ((v,), dtype, path + "post_attention_layernorm.weight"),
            # w_gate
            ((h, v), dtype, path + "block_sparse_moe.gate.weight")
        ]
        weights = init_weight_list(weight_specs, self.policy, self.offload_dst, self.env, self.config.torch_dtype)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf):
        w_ffn_ln, w_gate = weight_home.val
        dst = self.weight_load_dst
        dst2 = self.compute
        weight_read_buf.store((w_gate.smart_copy(dst, isMixtralWeight=True), w_ffn_ln.smart_copy(dst2, isMixtralWeight=True)))
        # weight_read_buf.store((w_gate.data.move_to(dst), w_ffn_ln.data.move_to(dst2)))
    

    def forward(self, hidden, weight_read_buf, cur_gate_buf, attention_mask, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_gate, donate[1]), (w_ffn_ln, donate[2])) = weight_read_buf.pop()
        else:
            ((w_gate, _), (w_ffn_ln, _)) = weight_read_buf.val
        h, expert_mask, routing_weights, selected_experts = self.compute.mixtral_gate(h, w_gate, w_ffn_ln, self.config, donate)
        hidden.val = h
        cur_gate_buf.store((expert_mask, routing_weights, selected_experts))

class MixtralSparseMLP:
    def __init__(self, config, env, policy, layer_idx):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = self.env.Ascend
        self.weight_load_dst = (self.compute.quantized_device if policy.HQQquantize_weight else self.compute)
        
        if isinstance(policy.offload_expert_weight, int):
            self.offload_dst = self.policy.offload_expert_weight
        else:
            self.offload_dst = self.policy.offload_expert_weight[self.layer_idx]
    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path, idx):
        hidden_dim, ffn_dim, dtype = self.config.hidden_size, self.config.intermediate_size, self.config.dtype
        path = os.path.join(path, f"layers.{self.layer_idx}.block_sparse_moe.experts.")
        weight_specs = [
            # w1
            ((ffn_dim, hidden_dim), dtype, path + f"{idx}.w1.weight"),
            # w2
            ((hidden_dim, ffn_dim), dtype, path + f"{idx}.w2.weight"),
            # w3
            ((ffn_dim, hidden_dim), dtype, path + f"{idx}.w3.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.offload_dst, self.env, self.config.torch_dtype)
        weight_home.store(weights)
    
    def load_weight(self, weight_home, weight_read_buf):
        w1, w2, w3 = weight_home.val
        dst = self.weight_load_dst
        weight_read_buf.store((w1.smart_copy(dst, isMixtralWeight=True), 
                               w2.smart_copy(dst, isMixtralWeight=True), 
                               w3.smart_copy(dst, isMixtralWeight=True)))
        if w1.device.device_type == DeviceType.QUANTIZE:
            ((w1, d1), (w2, d2), (w3, d3)) = weight_read_buf.val
            w1 = w1.device.dequantize(w1)
            w2 = w2.device.dequantize(w2)
            w3 = w3.device.dequantize(w3)
            weight_read_buf.val = ((w1, d1), (w2, d2), (w3, d3)) 
    
    def forward(self, hidden, weight_read_buf, routing_weights):
        ((w1, _), (w2, _), (w3, _)) = weight_read_buf.val
        h = self.compute.sparseMLP(hidden, w1, w2, w3, routing_weights)
        
        return h

class MixtralSparseMoe:
    def __init__(self, config, env, policy, layer_idx):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = self.env.Ascend
        self.hot_experts = None

        self.experts = {}
        for idx in range(config.num_local_experts):
            self.experts[f"expert_{idx}"] = MixtralSparseMLP(config, self.env, self.policy, layer_idx)
    
    def set_task(self, task):
        for idx in range(self.config.num_local_experts):
            self.experts[f"expert_{idx}"].set_task(task)

    def init_weight(self, weight_home, path):
        home = {}
        for idx in range(self.config.num_local_experts):
            home[f"expert_{idx}"] =  ValueHolder()
            self.experts[f"expert_{idx}"].init_weight(home[f"expert_{idx}"], path, idx)

        weight_home.store((home))
    
    def load_weight(self, weight_home, weight_read_buf, prefetch_expert_index: list = None):
        read_buf = {}
        home = weight_home.val
        if prefetch_expert_index is None:
            for idx in range(self.config.num_local_experts):
                read_buf[f"expert_{idx}"] = ValueHolder()
                home_expert = home[f"expert_{idx}"]
                self.experts[f"expert_{idx}"].load_weight(home_expert, read_buf[f"expert_{idx}"])
        else:
            for idx in prefetch_expert_index:
                read_buf[f"expert_{idx}"] = ValueHolder()
                home_expert = home[f"expert_{idx}"]
                self.experts[f"expert_{idx}"].load_weight(home_expert, read_buf[f"expert_{idx}"])

        weight_read_buf.store((read_buf))

    def load_weight_after_gate(self, weight_home, weight_after_gate_read_buf, index):
        read_buf = {}
        read_buf[f"expert_{index}"] = ValueHolder()
        home = weight_home.val

        home_expert = home[f"expert_{index}"]
        self.experts[f"expert_{index}"].load_weight(home_expert, read_buf[f"expert_{index}"])

        weight_after_gate_read_buf.store((read_buf))

    def forward(self, hidden, residual, weight_read_buf, weight_after_gate_read_buf, cur_gate_buf, transfer_expert_list):
        prefetch_experts = weight_read_buf.pop()
        mm=copy.deepcopy(hidden)
        # Get expert popularity
        if self.hot_experts is None:
            if self.policy.num_gpu_batches == 1:
                donate = [False] * 1
                h, donate[0] = hidden.val, True
                
                h = self.compute.MoE_computation(h, self.experts, prefetch_experts, cur_gate_buf[0], donate)
                hidden.val = h
                return
            else:
                results = {}
                for k in range(self.policy.num_gpu_batches):
                    donate = [False] * 1
                    h, donate[0] = hidden[k].val, True
                    
                    h = self.compute.MoE_computation(h, self.experts, prefetch_experts, cur_gate_buf[k], donate)
                    results[k] = h
        # Inference
        else:
            results = self.compute.mixtral_MoE(hidden, residual, self.experts, self.hot_experts, prefetch_experts, 
                                               weight_after_gate_read_buf, cur_gate_buf, transfer_expert_list, self.policy.num_gpu_batches)
            
        return results

class MixtralModel:
    def __init__(self, config: MixtralConfig, env: ExecutionEnv, path: str, policy: Policy):
        if isinstance(config, str):
            config = get_mixtral_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.expert_popularity = None

        self.layers = []
        self.layers.append(MixtralInputEmbedding(config, self.env, self.policy))
        for layer_idx in range(config.num_hidden_layers): 
            self.layers.append(MixtralAttention(config, self.env, self.policy, layer_idx))
            self.layers.append(MixtralTop2Gate(config, self.env, self.policy, layer_idx))
            self.layers.append(MixtralSparseMoe(config, self.env, self.policy, layer_idx))
        self.layers.append(MixtralOutputEmbedding(config, self.env, self.policy))

        self.num_layers = len(self.layers)

        dev_choices = [self.env.Ascend, self.env.cpu, self.env.disk]
        self.hidden_home = dev_choices[self.policy.offload_dense_weight]
        
        # CUDA streams
        self.load_weight_stream = ms.hal.Stream()
        self.load_expert_stream = ms.hal.Stream()
        self.load_cache_stream = ms.hal.Stream()
        self.store_cache_stream = ms.hal.Stream()

        self.init_all_weights()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        
        if not os.path.exists(check_path):
            download_mixtral_weights(self.config.name, self.path)

        print("Init weight of layers.")
        for j in tqdm(range(self.num_layers), desc="Initializing..."):
            self.layers[j].init_weight(self.weight_home[j], expanded_path)
    
    def get_prefetch_expert_index(self, j):
        if self.layers[j].layer_idx != 0 and self.pre_gate_buf is None:
            raise ValueError("layer_idx != 0, but pre_gate_buf is None!")
        
        if self.layers[j].layer_idx == 0:
            prefetch_expert_index = sorted(self.expert_popularity[0], key=self.expert_popularity[0].get, reverse=True)[:2]
        else:
            pre_index_counter = {i: 0 for i in range(8)}
            for gate_buf in self.pre_gate_buf:
                for top2_index in gate_buf.val:
                    for index in top2_index:
                        pre_index_counter[index] += 1
            
            expert_popularity = self.expert_popularity[self.layers[j].layer_idx]
            expert_counter = {i: 0 for i in range(8)}
            for index in pre_index_counter:
                get_top2_index = sorted(expert_popularity[index], key=expert_popularity[index].get, reverse=True)[:2]
                expert_counter[get_top2_index[0]] += pre_index_counter[index]
                expert_counter[get_top2_index[1]] += pre_index_counter[index]
            prefetch_expert_index = sorted(expert_counter, key=expert_counter.get, reverse=True)[:2]

        return prefetch_expert_index
    
    def load_weight(self, i, j, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with ms.hal.StreamCtx(self.load_weight_stream):
                if isinstance(self.layers[j], MixtralSparseMoe):
                    # warmup
                    if self.expert_popularity is None:
                        self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j])
                    # generate
                    else:
                        prefetch_expert_index = self.get_prefetch_expert_index(j)
                        self.layers[j].hot_experts = prefetch_expert_index
                        for index in prefetch_expert_index:
                            self.transfer_expert_list.append(index)
                        self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], prefetch_expert_index)
                else:
                    self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j])
        else:
            if isinstance(self.layers[j], MixtralSparseMoe):
                # warmup
                if self.expert_popularity is None:
                    self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j])
                # generate
                else:
                    prefetch_expert_index = self.get_prefetch_expert_index(j)
                    self.layers[j].hot_experts = prefetch_expert_index
                    for index in prefetch_expert_index:
                        self.transfer_expert_list.append(index)
                    self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], prefetch_expert_index)
            else:
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j])
        
        if isinstance(self.layers[j], MixtralTop2Gate) and self.expert_popularity is not None:
            self.load_weight(i, j+1)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        if isinstance(self.layers[j], MixtralAttention):
            self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return
            
        if not isinstance(self.layers[j], MixtralAttention):
            return
        # Load from cache_home to cache_read_buf
        if overlap:
            with ms.hal.StreamCtx(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return
        
        if not isinstance(self.layers[j], MixtralAttention):
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            # print("layer",(j,k))
            with ms.hal.StreamCtx(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        # if v:
        #     for x in v:
        #         x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        if isinstance(self.layers[j], MixtralTop2Gate) and self.expert_popularity:
            self.hidden[i][j][k] = copy.deepcopy(self.hidden[i][j-1][k])
        else:
            dst = self.layers[j].compute
            if j == 0:
                gpu_batch_size = self.policy.gpu_batch_size
                left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
                if i == 0:  # load from the input ids
                    val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                    val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
                else:  # load from the last generated token
                    pos = self.task.prompt_len + i
                    val = dst.allocate((gpu_batch_size, 1), np.int32)
                    val.load_from_np(self.output_ids[left:right, pos-1:pos])
            else:  # load from the last layer
                val = self.hidden[i][j-1][k].pop().move(dst)
            self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k=None):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        if isinstance(self.layers[j], MixtralSparseMoe) and self.num_gpu_batches > 1:
            for batch in range(self.num_gpu_batches):
                x = self.hidden[i][j][batch]
                if x.val:  # x may already be moved due to overlapping
                    x.val = x.val.move(self.hidden_home)
        else:
            # Store to hidden states buffers
            if j == self.num_layers - 1:  # store to output
                gpu_batch_size = self.policy.gpu_batch_size
                left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
                # ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
                ids = self.hidden[i][j][k].pop().data.numpy()
                pos = self.task.prompt_len + i
                if self.task.stop:
                    stopped = self.stopped[left:right]
                    self.output_ids[left:right, pos:pos+1] = np.where(stopped, self.config.pad_token_id, ids)
                    stopped[:] = np.logical_or(stopped, ids == self.task.stop)
                else:
                    self.output_ids[left:right, pos:pos+1] = ids
            else:  # move to home
                x = self.hidden[i][j][k]
                if x.val:  # x may already be moved due to overlapping
                    x.val = x.val.move(self.hidden_home)

    def load_weight_after_gate(self, i, j, k, overlap):
        _, _, selected_experts = self.cur_gate_buf[k].val

        expert_index_list = selected_experts.tolist()
        for index_list in expert_index_list:
            for index in index_list:
                if index not in self.transfer_expert_list:
                    self.transfer_expert_list.append(index)
                    if overlap:
                        with ms.hal.StreamCtx(self.load_expert_stream):
                            self.layers[j+1].load_weight_after_gate(self.weight_home[j+1], self.weight_after_gate_read_buf[j+1], index)
                    else:
                        self.layers[j+1].load_weight_after_gate(self.weight_home[j+1], self.weight_after_gate_read_buf[j+1], index)
        if k == self.num_gpu_batches - 1 and self.expert_popularity is None:
            self.all_gate_select[f"{i},{self.layers[j].layer_idx}"] = copy.deepcopy(self.cur_gate_buf)

    def update_expert_popularity(self, j):
        # warm up, transfer gate value only
        if self.expert_popularity is None:
            for k in range(self.num_gpu_batches):
                self.pre_gate_buf[k].clear()
                _, _, selected_experts = self.cur_gate_buf[k].pop()
                self.pre_gate_buf[k].store(selected_experts.tolist())
        # generate, update expert_popularity
        else:
            for k in range(self.num_gpu_batches):
                _, _, selected_experts = self.cur_gate_buf[k].pop()
                tmp_cur_indices = selected_experts.tolist()
                if self.layers[j].layer_idx == 0:
                    for top2_index in tmp_cur_indices:
                        for index in top2_index:
                            self.expert_popularity[0][index] += 1
                else:
                    tmp_pre_indices = self.pre_gate_buf[k].val
                    for tmp_pre_index, tmp_cur_index in zip(tmp_pre_indices, tmp_cur_indices):
                        for pre_top2 in tmp_pre_index:
                            for cur_top2 in tmp_cur_index:
                                layer_index = self.layers[j].layer_idx
                                self.expert_popularity[layer_index][pre_top2][cur_top2] += 1
                self.pre_gate_buf[k].clear()
                self.pre_gate_buf[k].store(tmp_cur_indices)

    def compute_layer(self, i, j, k=None, overlap=True):
        if isinstance(self.layers[j], MixtralSparseMoe):
            if self.num_gpu_batches == 1:
                self.layers[j].forward(self.hidden[i][j][0], self.hidden[i][j-2], self.weight_read_buf[j], 
                    self.weight_after_gate_read_buf[j], self.cur_gate_buf, self.transfer_expert_list)
            else:
                results = self.layers[j].forward(self.hidden[i][j-1], self.hidden[i][j-2], self.weight_read_buf[j], 
                    self.weight_after_gate_read_buf[j], self.cur_gate_buf, self.transfer_expert_list)
                for k in range(self.num_gpu_batches):
                    self.hidden[i][j][k].store(results[k])
                    self.hidden[i][j-1][k].clear()
                    self.hidden[i][j-2][k].clear()
        elif isinstance(self.layers[j], MixtralAttention):
            self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k], self.weight_read_buf[j], 
                self.attention_mask[k], self.cache_write_buf[j][k], self.position_ids[k], i, k)
        else:
            self.layers[j].forward(self.hidden[i][j][k], self.weight_read_buf[j], 
                                   self.cur_gate_buf[k], self.attention_mask[k], k)

        if i == 0 and self.execute_gen_len != 1:
            # current_memory = torch.cuda.memory_allocated(self.env.gpu.dev)
            current_memory = 0
            self.memory_usage.append(current_memory/GB)

        # load_weight_after_gate
        if isinstance(self.layers[j], MixtralTop2Gate) and self.cur_gate_buf[k].val is not None:
            self.load_weight_after_gate(i, j, k, overlap)

        if isinstance(self.layers[j], MixtralSparseMoe) and self.cur_gate_buf[k].val is not None:
            self.update_expert_popularity(j)
        

    def sync(self, stream=None):
        if stream is None:
            ms.hal.synchronize()
        else:
            self.env.disk.synchronize()
            stream.synchronize()

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            self.position_ids[k] = mask.val.data.cumsum(-1) - 1
            self.position_ids[k] .masked_fill(mask.val.data == 0, 1)
            self.position_ids[k] = self.position_ids[k][:, -1:]

            return
        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = self.env.Ascend
        val = attention_compute.allocate((self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)
        # self.position_ids[k] = val.data.long().cumsum(-1) - 1
        self.position_ids[k] = val.data.cumsum(-1) - 1
        self.position_ids[k].masked_fill(val.data == 0, 1)
    
    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
    
    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.prompt_len = prompt_len
        self.execute_gen_len = task.gen_len
        self.memory_usage = []

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs,dtype=np.int32)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)
        

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.weight_after_gate_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        # Intermediate tensors
        # The following buffers store values used for the i-th token, j-th layer, k-th gpu batch.
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            self.weight_after_gate_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()

        # init position_ids
        self.position_ids = array_1d(num_gpu_batches, ValueHolder)
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
        self.pre_gate_buf = array_1d(num_gpu_batches, ValueHolder)
        self.cur_gate_buf = array_1d(num_gpu_batches, ValueHolder)
        self.all_gate_select = {}
        self.transfer_expert_list = []

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)

        # Generate
        if not overlap:
            # No overlap
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if num_gpu_batches == 1:
                self.generation_loop_overlap_single_batch()
            else:
                self.generation_loop_overlap_multi_batch()

        file_name = "mixtral/experiment/gpu_time"+str(num_gpu_batches)+"*"+str(gpu_batch_size)+"_offload.log"
        with open(file_name, "a") as fout:
            for usage in self.memory_usage:
                fout.write(str(usage) + '\n')
        
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()
        self.expert_popularity = None

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_single_batch_our(self):
        # Prologue
        self.load_weight(0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            if i == 0:
                past_key_values_length = 0
                seq_length = self.prompt_len
            else:
                past_key_values_length = self.prompt_len + i - 1
                seq_length = 1
            for m in range(self.num_gpu_batches):
                self.position_ids[m] = ms.ops.arange(past_key_values_length, past_key_values_length + seq_length,
                                                    dtype=ms.int64)
                self.position_ids[m] = self.position_ids[m].unsqueeze(0).view(-1, seq_length)
            for j in range(self.num_layers):
                self.load_weight(i, j+1)
                if isinstance(self.layers[j], MixtralTop2Gate):
                    self.load_hidden(i, j, 0)
                    self.compute_layer(i, j, 0)
                elif isinstance(self.layers[j], MixtralSparseMoe):
                    self.compute_layer(i, j)
                    self.store_hidden(i, j)
                    # self.load_hidden(i, j+1, 0)
                    self.load_cache(i, j+1, 0)
                else:
                    self.load_cache(i, j+1, 0)
                    self.load_hidden(i, j, 0)
                    self.compute_layer(i, j, 0)
                    self.store_cache(i, j, 0)
                    if not isinstance(self.layers[j], MixtralAttention):
                        self.store_hidden(i, j, 0)
                    self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        self.load_weight(0, 0)
        self.load_hidden(0, 0, 0)
        self.sync()
        
        # Generate
        for i in tqdm(range(self.execute_gen_len), desc="Generating..."):
            start_time = time.time()
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            
            for j in range(self.num_layers):
                # print(ms.hal.max_memory_allocated("Ascend")/1024/1024)
                if (not isinstance(self.layers[j], MixtralTop2Gate)) or (self.expert_popularity is None):
                    self.load_weight(i, j+1)
                if isinstance(self.layers[j], MixtralTop2Gate):
                    for k in range(self.num_gpu_batches):
                        if k != self.num_gpu_batches - 1:
                            self.load_hidden(i, j, k+1)
                        
                        self.compute_layer(i, j, k=k)
                elif isinstance(self.layers[j], MixtralSparseMoe):
                    self.load_cache(i, j, self.num_gpu_batches)
                    self.compute_layer(i, j)
                    self.store_hidden(i, j)
                    self.load_hidden(i, j, self.num_gpu_batches)
                else:
                    for k in range(self.num_gpu_batches):
                        self.load_hidden(i, j, k+1)
                        self.sync(stream=self.load_cache_stream)
                        self.load_cache(i, j, k+1)

                        
                        self.compute_layer(i, j, k=k)
                        self.sync(stream=self.store_cache_stream)
                        # self.store_cache(i, j, k-1)
                        self.store_cache(i, j, k)
                        if not isinstance(self.layers[j], MixtralAttention):
                            self.store_hidden(i, j, k=k)
                self.sync(stream=self.load_weight_stream)
            timers("generate").stop()
            end_time = time.time()
        
def convert_keys_to_int(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            value = convert_keys_to_int(value)
        new_dict[int(key)] = value
    return new_dict
    
def get_expert_popularity(all_gate_select, num_hidden_layers, check_popularity, num_local_experts, num_gpu_batches, gen_len):
    expert_popularity = {}
    for i in range(num_hidden_layers):
        expert_popularity[i] = {}
        for j in range(num_local_experts):
            if i == 0:
                expert_popularity[i][j] = 0
            else:
                expert_popularity[i][j] = {}
                for k in range(num_local_experts):
                    expert_popularity[i][j][k] = 0

    with open(check_popularity, "w+") as file:
        for gen_num in range(gen_len):
            for layer in range(num_hidden_layers):
                for batch in range(num_gpu_batches):
                    gate_record = all_gate_select[f'{gen_num},{layer}'][batch].val
                    _, _, expert_selected = gate_record
                    if layer == 0:
                        for expert_indexs in expert_selected:
                            for expert_index in expert_indexs.tolist():
                                expert_popularity[0][expert_index] += 1
                    else:
                        pre_gate_record = all_gate_select[f'{gen_num},{layer-1}'][batch].val
                        _, _, pre_expert_selected = pre_gate_record
                        for expert_indexs, pre_expert_indexs in zip(expert_selected, pre_expert_selected):
                            for pre_expert_index in pre_expert_indexs.tolist():
                                for expert_index in expert_indexs.tolist():
                                    expert_popularity[layer][pre_expert_index][expert_index] += 1
        # print(expert_popularity)
        file.write(json.dumps(expert_popularity))

    return expert_popularity


def get_filename(args):
    offload = "offload-"
    for i in range(len(args.offload)):
        offload += str(args.offload[i]) + "-"
    filename = f"Output-mixtral-8x22b-ngb{args.num_gpu_batches}-" \
               f"gbs{args.gpu_batch_size}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-{offload}"

    if args.attn_sinkToken:
        filename += "attn-sinkToken"
    else:
        filename += "gpu-cache"
    if args.HQQ_quantize:
        filename += "-quantw"
    if args.overlap == False:
        filename += "no_overlap"

    return filename

def empty_folder(offload_dir):
    import shutil
    shutil.rmtree(offload_dir)
    os.makedirs(offload_dir)

def get_line_from_dataset(dataset, line):
    # there are some topics and None in dataset
    if len(dataset['train'][line]['text']) < 128:
        line += 1
        return get_line_from_dataset(dataset, line)
    return dataset['train'][line]['text']

def get_warmup_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Hey, are you conscious? Can you talk to me?"]
    input_ids = tokenizer(prompts, padding="max_length", max_length=prompt_len, return_tensors="pt").input_ids[:,:prompt_len]
    return (input_ids[0],) * num_prompts

# get inputs for generating a expert_popularity table
def get_ep_inputs(prompt_len, num_prompts, tokenizer):
    if not os.path.exists('dataset/wikitext-2-v1'):
        os.makedirs('dataset/wikitext-2-v1')
        # load from HF
        dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
        dataset.save_to_disk('dataset/wikitext-2-v1')
    else:
        # load from local
        dataset = datasets.load_from_disk('dataset/wikitext-2-v1')
    get_line = []
    for _ in range(num_prompts):
        get_line.append(random.randint(0, 36718-1))
    inputs = ()
    for line in get_line:
        input_ids = tokenizer(get_line_from_dataset(dataset, line), padding="max_length", truncation=True, max_length=prompt_len).input_ids
        inputs = inputs + (input_ids,)
    return inputs

def get_inputs(prompt_len, num_prompts, tokenizer):
    if not os.path.exists('dataset/wikitext-103-v1'):
        os.makedirs('dataset/wikitext-103-v1')
        # load from HF
        dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')
        dataset.save_to_disk('dataset/wikitext-103-v1')
    else:
        # load from local
        dataset = datasets.load_from_disk('dataset/wikitext-103-v1')
    get_line = []
    for _ in range(num_prompts):
        get_line.append(random.randint(0, 1801350-1))

    inputs = ()
    for line in get_line:
        input_ids = tokenizer(get_line_from_dataset(dataset, line), padding="max_length", truncation=True, max_length=prompt_len).input_ids
        inputs = inputs + (input_ids,) 
    return inputs

def run_umoe(args):
    print(f"<Start to run>: {args.model}")
    if args.model == "mistralai/Mixtral-8x7B-v0.1":
        tokenizer = AutoTokenizer.from_pretrained("mixtral/Tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "mistralai/Mixtral-8x22B-v0.1":
        tokenizer = AutoTokenizer.from_pretrained("mixtral/Tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"can't find {args.model}")
    
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len = args.prompt_len, args.gen_len
    gpu = TorchDevice("cuda")
    Ascend = TorchDevice("Ascend")
    cpu = TorchDevice("CPU")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,Ascend=Ascend)

    maxtral_config = get_mixtral_config(args.model)
    if args.manual_offload[0] == -1:
        auto_offload = maxtral_config.memory_cost_mixtral(num_prompts, prompt_len + gen_len, args.device, args.HQQ_quantize)
        policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    auto_offload[0], auto_offload[1], auto_offload[2],
                    args.overlap, args.pin_weight, args.attn_sinkToken, args.HQQ_quantize,
                    HQQConfig(num_bits=4, group_size=64, axis=1), HQQConfig(num_bits=4, group_size=16, axis=0))
        print("policy: ", policy)
    else:
        policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                        args.manual_offload[0], args.manual_offload[1], args.manual_offload[2],
                        args.overlap, args.pin_weight, args.attn_sinkToken, args.HQQ_quantize,
                        HQQConfig(num_bits=4, group_size=64, axis=1), HQQConfig(num_bits=4, group_size=16, axis=0))
    
    if args.HQQ_config[0] != 0:
        policy.HQQ_attn_config = HQQConfig(num_bits=args.HQQ_config[0], group_size=args.HQQ_config[1], axis=1)
        policy.HQQ_expert_config = HQQConfig(num_bits=args.HQQ_config[2], group_size=args.HQQ_config[3], axis=0)

    maxtral_config = get_mixtral_config(args.model)
    cache_size = maxtral_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = maxtral_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {maxtral_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    model = MixtralModel(maxtral_config, env, args.path, policy)
    try:
        if args.model == "mistralai/Mixtral-8x7B-v0.1":
            check_popularity = "mixtral/expert_path.json"
        else:
            check_popularity = "mixtral/expert_path_8x22.json"
        if not os.path.exists(check_popularity):
            print("Warmup and get expert popularity.")
            warmup_inputs = get_ep_inputs(16, num_prompts, tokenizer)
            output_ids = model.generate(warmup_inputs, max_new_tokens=args.gen_len)
            expert_popularity = get_expert_popularity(model.all_gate_select, maxtral_config.num_hidden_layers, check_popularity,
                                                      maxtral_config.num_local_experts, args.num_gpu_batches, args.gen_len)
        else:
            with open(check_popularity, 'r') as file:
                expert_popularity = json.load(file)
            expert_popularity = convert_keys_to_int(expert_popularity)
            print("Expert popularity already exists. Just warm up.")
            warmup_inputs = get_warmup_inputs(10, num_prompts, tokenizer)
            output_ids = model.generate(warmup_inputs, max_new_tokens=1)
        model.expert_popularity = expert_popularity
        print("benchmark - generate")
        inputs = get_inputs(prompt_len, num_prompts, tokenizer)
        inputs = get_warmup_inputs(prompt_len, num_prompts, tokenizer)
        print("inputs",inputs)
        timers("generate").reset()
        output_ids = model.generate(inputs, max_new_tokens=args.gen_len)
        costs = timers("generate").costs

        # Log output
        prefill_latency = costs[0]
        prefill_throughput = num_prompts * prompt_len / prefill_latency
        decode_latency = sum(costs[1:])
        decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
        num_generated_tokens = num_prompts * gen_len
        total_latency = prefill_latency + decode_latency
        total_throughput = num_generated_tokens / total_latency
        _, gpu_peak_mem = Ascend.mem_stats()
        # _, cpu_peak_mem = cpu.mem_stats()
        cpu_peak_mem=0

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        print("inputs: ", inputs)
        # print("outputs: ", outputs)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in range(len(outputs)):
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
            # if i == 1:
        print(show_str)
        # if args.store_output:
        import os
        os.makedirs('output', exist_ok=True)
        with open('output/outputs.log', "a+") as out_file:
            out_file.write(show_str)
    finally:
        env.close_copy_threads()
