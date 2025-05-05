import math
from typing import List, Optional, Tuple, Union, Dict
import copy
import os
from collections import Counter
import sys
from tqdm import tqdm
import mindnlp.core.nn.functional as F

import numpy as np
import time
# from safetensors.torch import load_file
from mindnlp.core.serialization import safe_load_file as load_file

from mindnlp.common.activations import ACT2FN
from mindnlp.transformers.cache_utils import Cache, DynamicCache, StaticCache
from mindnlp.transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .configuration_qwen import Qwen2MoeConfig, get_Qwen_config
from weights_download import download_Qwen_weights
from quantizer import dequantize, quantize
from expert_ARC_cache import ARC_Cache
from utils import memory_cost_qwen
import mindspore as ms
from mindnlp.core import nn, ops

prefetch_stream = ms.hal.Stream()
load_stream = ms.hal.Stream()
quan_expert = ms.hal.Stream()

def dict_move(obj,device):
    dict={}
    for key in obj:
        dict[key]=obj[key].move_to(device)
    return dict

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: ms.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: ms.dtype,
    min_dtype: float,
    cache_position: ms.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = ms.ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = ms.ops.triu(causal_mask, diagonal=1)
        causal_mask *= ms.ops.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand((batch_size, 1, -1, -1))
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class Qwen2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        #问题：一开始就是16，还是后面一起转成fp16
        self.weight = nn.Parameter(ms.ops.ones(hidden_size,dtype=ms.float16))
        self.variance_epsilon = eps
    def init_weights(self, path):
        self.weight.data.copy_(load_file(path)['tensor'])

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ms.ops.rsqrt(variance + self.variance_epsilon)
        a=self.weight * hidden_states.to(input_dtype)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2MoeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ms.ops.arange(0, self.dim, 2, dtype=ms.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=ms.float16
        )

    def _set_cos_sin_cache(self, seq_len, dtype, device=None):
        self.max_seq_len_cached = seq_len
        t = ms.ops.arange(self.max_seq_len_cached, dtype=ms.int64).type_as(self.inv_freq)

        freqs = ms.ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ms.ops.cat((freqs, freqs), axis=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            # self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)


        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ms.ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2MoeAttention(nn.Module):

    def __init__(self, config: Qwen2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2MoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def init_weights(self, path):
        path = path + f"/original/model.layers.{self.layer_idx}.self_attn."
        self.q_proj.weight.data.copy_(load_file(path+"q_proj.weight")['tensor'])
        self.q_proj.bias.data.copy_(load_file(path + "q_proj.bias")['tensor'])
        self.k_proj.weight.data.copy_(load_file(path + "k_proj.weight")['tensor'])
        self.k_proj.bias.data.copy_(load_file(path + "k_proj.bias")['tensor'])
        self.v_proj.weight.data.copy_(load_file(path + "v_proj.weight")['tensor'])
        self.v_proj.bias.data.copy_(load_file(path + "v_proj.bias")['tensor'])
        self.o_proj.weight.data.copy_(load_file(path + "o_proj.weight")['tensor'])

    def forward(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = ops.transpose(query_states.view(bsz, q_len, self.num_heads, self.head_dim), 1, 2)
        key_states = ops.transpose(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)
        value_states = ops.transpose(value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, ops.transpose(key_states, 2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = ops.transpose(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)



        return attn_output, past_key_value


class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, layer_idx, is_shared):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if self.layer_idx == 0:
            self.quan_bit = 0
        else:
            self.quan_bit = self.config.quan_map[layer_idx]
        self.is_shared = is_shared
        
        self.gate = None
        self.up = None
        self.down = None
        self.act_fn = ACT2FN[config.hidden_act]

    def init_weights(self, path, idx=None, num_in_mem=None):
        # shared_expert
        if idx is None:
            path = path + f"/original/model.layers.{self.layer_idx}.mlp.shared_expert."
            self.gate_proj_path = path + f"gate_proj.weight"
            self.up_proj_path = path + f"up_proj.weight"
            self.down_proj = path + f"down_proj.weight"

            self.gate = load_file(self.gate_proj_path)['tensor']
            self.up = load_file(self.up_proj_path)['tensor']
            self.down = load_file(self.down_proj)['tensor']
        else:
            self.idx = idx
            self.weight_path = {}
            # locate fp16 weights
            self.weight_path[0] = path + f"/original/model.layers.{self.layer_idx}.mlp.experts.{idx}.weight"
            # locate int4/2 weights
            self.weight_path[4] = path + f"/quantized/int4/model.layers.{self.layer_idx}.mlp.experts.{idx}.weight"
            self.weight_path[2] = path + f"/quantized/int2/model.layers.{self.layer_idx}.mlp.experts.{idx}.weight"

            init_device = 'CPU'
            self.gate_cpu = {}
            self.up_cpu = {}
            self.down_cpu = {}
            if self.quan_bit == 0:
                # weight = load_file(self.weight_path[0], device=init_device)
                weight = load_file(self.weight_path[0])
                self.gate_cpu[0] = weight.pop("gate").move_to(init_device)
                self.up_cpu[0] = weight.pop("up").move_to(init_device)
                self.down_cpu[0] = weight.pop("down").move_to(init_device)

                self.gate = self.gate_cpu[self.quan_bit]
                self.up = self.up_cpu[self.quan_bit]
                self.down = self.down_cpu[self.quan_bit]

                self.gate_cpu[0] = None
                self.up_cpu[0] = None
                self.down_cpu[0] = None
                weight=None

            weight = load_file(self.weight_path[4])
            self.gate_cpu[4] = dict_move(self.extract_keys('gate', weight), init_device)
            self.up_cpu[4] = dict_move(self.extract_keys('up', weight),init_device)
            self.down_cpu[4] = dict_move(self.extract_keys('down', weight),init_device)

            weight = load_file(self.weight_path[2])
            self.gate_cpu[2] = dict_move(self.extract_keys('gate', weight),init_device)
            self.up_cpu[2] = dict_move(self.extract_keys('up', weight), init_device)
            self.down_cpu[2] = dict_move(self.extract_keys('down', weight), init_device)

            if num_in_mem is not None and idx < num_in_mem:
                if self.quan_bit != 0:
                    self.gate = {k: v for k, v in self.gate_cpu[self.quan_bit].items()}
                    self.up = {k: v for k, v in self.up_cpu[self.quan_bit].items()}
                    self.down = {k: v for k, v in self.down_cpu[self.quan_bit].items()}


    
    
    def extract_keys(self, prefix, weight):
        return {
            'nbits': weight.pop(f'{prefix}_nbits'),
            'shape': weight.pop(f'{prefix}_shape'),
            'W_q': weight.pop(f'{prefix}'),
            'scale': weight.pop(f'{prefix}_scale'),
            'zero': weight.pop(f'{prefix}_zero')
        }
    
    def load_from_cpu(self, weight):
        return {
            'nbits': weight['nbits'],
            'shape': weight['shape'],
            'W_q': weight['W_q'].move_to("Ascend", blocking=False),
            'scale': weight['scale'].move_to("Ascend", blocking=False),
            'zero': weight['zero'].move_to("Ascend", blocking=False)
        }

    def load_weights(self, is_now=False, nbit=None):
        quan_bit = self.quan_bit if nbit is None else nbit

        if is_now:
            self.gate = self.load_from_cpu(self.gate_cpu[quan_bit])
            self.up = self.load_from_cpu(self.up_cpu[quan_bit])
            self.down = self.load_from_cpu(self.down_cpu[quan_bit])
        else:
            with ms.hal.StreamCtx(prefetch_stream):
                self.gate = self.load_from_cpu(self.gate_cpu[quan_bit])
                self.up = self.load_from_cpu(self.up_cpu[quan_bit])
                self.down = self.load_from_cpu(self.down_cpu[quan_bit])

    def dequan_experts(self):
        if not self.is_shared and self.quan_bit != 0:
            self.gate = dequantize(self.gate)
            self.up = dequantize(self.up)
            self.down = dequantize(self.down)
    
    def quan_experts(self):
        if self.quan_bit != 0:
            self.gate = self.load_from_cpu(self.gate_cpu[self.quan_bit])
            self.up = self.load_from_cpu(self.up_cpu[self.quan_bit])
            self.down = self.load_from_cpu(self.down_cpu[self.quan_bit])

    def clear(self):
        self.gate = None
        self.up = None
        self.down = None

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down) 


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.layer_idx = layer_idx
        self.device = config.device
        self.num_in_mem = int(self.num_experts - config.offload_map[layer_idx])

        self.arc_cache = ARC_Cache(self.num_in_mem)

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # self.gate_cpu = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, layer_idx, False) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen2MoeMLP(config, layer_idx, True)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
    
    def init_weights(self, path):
        gate_path = path + f"/original/model.layers.{self.layer_idx}.mlp.gate.weight"
        self.gate.weight.data.copy_(load_file(gate_path)['tensor'])
        shared_expert_gate_path = path + f"/original/model.layers.{self.layer_idx}.mlp.shared_expert_gate.weight"
        self.shared_expert_gate.weight.data.copy_(load_file(shared_expert_gate_path)['tensor']) 
        for idx in range(self.num_experts):
            self.experts[idx].init_weights(path, idx, self.num_in_mem)
        self.shared_expert.init_weights(path)
    
    def load_weights(self, idx, is_now=False, int2_experts=None):
        # load_stream
        if isinstance(idx, int):
            if self.arc_cache.is_evicted(idx):
                self.experts[idx].load_weights(is_now=is_now, nbit=2)
        # prefetch_stream
        else:
            for i in idx:
                if self.arc_cache.is_evicted(i):
                    nbit = 2 if i in int2_experts else 4
                    self.experts[i].load_weights(is_now=is_now, nbit=nbit)

    def post_comp(self, expert_idx):
        if self.num_in_mem == 0:
            self.experts[expert_idx].clear()
        elif self.layer_idx != 0:
            if self.arc_cache.is_evicted(expert_idx):
                self.experts[expert_idx].clear()
            else:
                
                with ms.hal.StreamCtx(quan_expert):
                    self.experts[expert_idx].quan_experts()

    def forward(self, hidden_states: ms.Tensor, prefetch_expert_idx) -> ms.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # gate comp.
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=ms.float16)
        routing_weights, selected_experts = ms.ops.topk(routing_weights, self.top_k, dim=-1)
            
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keep_dims=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # final_hidden_states = ms.ops.zeros(
        #     (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        # )
        final_hidden_states = ms.ops.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype
        )

        expert_mask = ms.ops.one_hot(selected_experts.to(ms.int64), self.num_experts).permute(2, 1, 0)

        # load in time.
        load_experts = []
        expert_index = selected_experts.view(-1).tolist()

        # if prefetch_expert_idx is None:
        if self.layer_idx == 0 or prefetch_expert_idx is None:
            prefetch_expert_idx = list(set(expert_index))
            evicted_list = []
        else:
            
            with ms.hal.cuda.StreamCtx(load_stream): 
                freq_counter = Counter(expert_index)
                freq_counter = [item[0] for item in sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)]
                for idx in freq_counter:
                    if idx not in prefetch_expert_idx:
                        self.load_weights(idx)
                        load_experts.append(idx)

            if self.num_in_mem != 0:
                evicted_list = self.arc_cache.update_list(expert_index)
                for idx in evicted_list:
                    self.experts[idx].clear()

        # experts comp.
        # comp. experts which had been prefetched
        for expert_idx in prefetch_expert_idx:
            # if expert_idx in expert_index, means it isn't paticipate in comp.
            if expert_idx not in expert_index:
                if self.arc_cache.is_evicted(expert_idx):
                    self.experts[expert_idx].clear()
                continue

            self.experts[expert_idx].dequan_experts()

            expert_layer = self.experts[expert_idx]
            idx, top_x = ms.mint.nonzero(expert_mask[expert_idx], as_tuple=True)

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
            self.post_comp(expert_idx)
        
        # comp. experts which loaded in time.
        # load_stream.synchronize()
        if len(load_experts) > 0:
            for expert_idx in load_experts:
                self.experts[expert_idx].dequan_experts()

                expert_layer = self.experts[expert_idx]
                idx, top_x = ops.nonzero(expert_mask[expert_idx], as_tuple=True)

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states.to(hidden_states.dtype))
                self.post_comp(expert_idx)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # self.self_attn = Qwen2MoeSdpaAttention(config, layer_idx)
        self.self_attn = Qwen2MoeAttention(config, layer_idx)

        self.mlp = Qwen2MoeSparseMoeBlock(config, layer_idx)

        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size,  eps=config.rms_norm_eps)
        self.next_gate_cpu = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def init_weights(self, path):
        self.self_attn.init_weights(path)
        self.mlp.init_weights(path)
        input_ln_path = path + f"/original/model.layers.{self.layer_idx}.input_layernorm.weight"
        post_ln_path = path + f"/original/model.layers.{self.layer_idx}.post_attention_layernorm.weight"
        self.input_layernorm.init_weights(input_ln_path)
        self.post_attention_layernorm.init_weights(post_ln_path)

        if self.layer_idx < self.config.num_hidden_layers - 1:
            gate_path = path + f"/original/model.layers.{self.layer_idx+1}.mlp.gate.weight"
            self.next_gate_cpu.weight.data.copy_(load_file(gate_path)['tensor'])

    def predict(self, hidden_states):
        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.next_gate_cpu(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=ms.float16)
        # _, selected_experts = torch.topk(routing_weights, self.mlp.top_k, dim=-1)
        _, selected_experts = ms.ops.topk(routing_weights, 5, dim=-1)
    
        return selected_experts

    def forward(
        self, hidden_states, attention_mask = None, position_ids = None, past_key_value = None,
        cache_position = None, prefetch_expert_list = None, next_layer = None
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # not prefetch only when (not offload) and (not quantize)
        next_prefetch_expert_list = None
        if self.layer_idx < self.config.num_hidden_layers-1 and self.config.offload_map[self.layer_idx+1] != 0:
            
            with ms.hal.StreamCtx(load_stream):
                # hidden_cpu = hidden_states.clone().to('cpu', non_blocking=True)
                hidden_cpu = hidden_states.clone()
                next_prefetch_expert_list = self.predict(hidden_cpu).view(-1).tolist()
                next_prefetch_expert_list = Counter(next_prefetch_expert_list)
                most_common_items = next_prefetch_expert_list.most_common()
                next_prefetch_expert_list = [item[0] for item in most_common_items]

                # determine which experts to int2
                next_prefetch_expert_dict = dict(most_common_items)
                value_sum = sum(next_prefetch_expert_dict.values())
                target_sum = value_sum * 0.3
                current_sum = 0
                int2_experts = []
                for key, value in reversed(next_prefetch_expert_dict.items()):
                    if current_sum + value > target_sum:
                        break
                    current_sum += value
                    int2_experts.append(key)
                next_layer.mlp.load_weights(next_prefetch_expert_list, int2_experts=int2_experts)
                del hidden_cpu

        hidden_states = self.mlp(hidden_states, prefetch_expert_list)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, next_prefetch_expert_list)

        return outputs


class Qwen2MoeModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def init_weights(self, path):
        # for i in range(self.config.num_hidden_layers):
        for i in tqdm(range(self.config.num_hidden_layers), desc="Init."):
            self.layers[i].init_weights(path)
        
        ln_path = path + "/original/model.norm.weight"
        embed_path = path + "/original/model.embed_tokens.weight"
        self.norm.init_weights(ln_path)
        self.embed_tokens.weight.data.copy_(load_file(embed_path)['tensor'])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        use_legacy_cache = False
        if not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            
            cache_position = ms.ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])
            
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = None
        next_prefetch_expert_list = None

        # for decoder_layer in self.layers:
        for i in range(self.config.num_hidden_layers):
            layer_outputs = self.layers[i](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                prefetch_expert_list=next_prefetch_expert_list,
                next_layer=self.layers[i+1] if i<self.config.num_hidden_layers-1 else None
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]
            next_prefetch_expert_list = layer_outputs[2]
            
        hidden_states = self.norm(hidden_states)

        next_cache = None
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        return tuple(
            v
            for v in [hidden_states, next_cache]
            if v is not None
        )

    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Cache,
    ):

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype = input_tensor.dtype
        min_dtype = float(ops.finfo(dtype).min)
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, ms.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask


class Qwen2MoeForCausalLM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.config = get_Qwen_config(args.model)
        self.config.device = eval(args.device)
        self.device = args.device
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.early_stopping = args.early_stopping
        self.path = args.path
        self.config._attn_implementation = "sdpa"

        (self.config.offload_map, self.config.quan_map) = memory_cost_qwen(self.config, args.memory_budget)
        # print("offload: ",self.config.offload_map)
        # print("quan_map: ",self.config.quan_map)

        self.model = Qwen2MoeModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.num_experts = self.config.num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok

        self.init_weights()

    def init_weights(self):
        expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, "qwen1.5-moe-a2.7b")))
        check_path = os.path.join(expanded_path, "original/lm_head.weight")
        if not os.path.exists(check_path):
            # print("didn't prepare weights!")
            # assert False
            download_Qwen_weights("Qwen/Qwen1.5-MoE-A2.7B", self.path)

        self.model.init_weights(expanded_path)
        self.lm_head.weight.data.copy_(load_file(check_path)['tensor'])

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple]:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # output = (logits,) + outputs[1:]
        return logits, outputs[1]
    
    def generate(self, input_ids, attention_mask=None, expriment_mode=None):
        prefill_time = 0
        if expriment_mode == "decoding":
            prefill_start_time = time.time()
        seq_len = input_ids.shape[1]
        past_key_values = DynamicCache()
        if attention_mask is None:
            attention_mask = ms.ops.ones((1, seq_len),dtype=ms.int32)
        position_ids = ms.ops.arange(0, seq_len).unsqueeze(0)
        cache_position = ms.ops.arange(0, seq_len)

        for i in tqdm(range(64), desc="Infer."):
            logits, past_key_values = self.forward(input_ids=input_ids, 
                                                attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                past_key_values=past_key_values,
                                                cache_position=cache_position,
                                                )
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            input_ids = ms.ops.argmax(logits, dim=-1)
            if i == 0:
                output = copy.deepcopy(input_ids[:, -1:])
            else:
                output = ms.ops.cat((output, input_ids), axis=1)

            input_ids = input_ids[:, -1:]
            # if self.early_stopping and i > self.min_length and input_ids.item() == self.config.eos_token_id:
            #     return output
            if input_ids.item() == self.config.eos_token_id:
                return (output, prefill_time)
            
            if i == 0:
                if expriment_mode == "prefill":
                    return (output, None)
                elif expriment_mode == "decoding":
                    prefill_time = time.time() - prefill_start_time
            
            # prepare for next decoding.
            position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)
            cache_position = (cache_position[-1] + 1).unsqueeze(-1)

            attention_mask = ms.ops.cat([attention_mask.to(ms.int32), ms.ops.ones((1, 1),dtype=ms.int32)], axis=-1)

        return (output, prefill_time)
