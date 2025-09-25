from enum import Enum, auto
from itertools import count
import os
import queue
import shutil
import threading
from typing import Tuple
import math
import torch
import torch.nn.functional as F
import numpy as np
from ml_dtypes import bfloat16
import mindspore as ms
from utils import (GB, cpu_mem_stats, vector_gather, np_dtype_to_torch_dtype, torch_dtype_to_np_dtype, torch_dtype_to_num_bytes)
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import Tensor, Parameter, mint
from mindspore.common import np_dtype

counttt=0
countttgate=0
TorchHQQDevice = None
global_cpu_device = None
global_disk_device = None
sliding_window = 256
sink_num = 2
count_out_emb=0

def fix_recursive_import():
    global TorchHQQDevice, global_cpu_device
    import HQQ_quantizer
    TorchHQQDevice = HQQ_quantizer.TorchHQQDevice

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    QUANTIZE = auto()
    Ascend=auto()

    @staticmethod
    def convert(name):
        if name == "CPU":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "quantized":
            return DeviceType.QUANTIZE
        elif name == "Ascend":
            return DeviceType.Ascend
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and quantized tensors on
        GPUs, CPUs and disks.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchHQQDevice, self.data is (data, scale, zero, hqq_config)
      type: Tuple[TorchTensor, dict]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        # if isinstance(data, Tensor):
        #     assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.device = device
        if device == 'CPU':
            self.data = data.move_to('CPU')
        else:
            self.data = data
        

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array, torch_dtype=None):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.QUANTIZE:
                tmp = Tensor.from_numpy(np_array)
                if torch_dtype is not None and (self.dtype != torch_dtype or tmp.dtype != torch_dtype):
                    tmp = tmp.to(torch_dtype)
                    self.dtype = torch_dtype
                tmp = global_cpu_device.quantized_device.quantize(tmp, self.data[3])
                general_copy(self, None, tmp, None)
            else:
                
                if torch_dtype is not None:
                    # self.dtype = torch_dtype
                    
                    ops.assign(self.data, Tensor.from_numpy(np_array).to(self.data.dtype))
                    
                else:
                    ops.assign(self.data, Tensor.from_numpy(np_array))
        if self.device.name=="CPU":
                self.data=self.data.move_to("CPU")

    def load_from_np_file(self, filename, torch_dtype):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename), torch_dtype)

    def copy(self, dst, isMixtralWeight=False, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices) + self.shape[len(src_indices):]
        else:
            shape = tuple(self.shape)

        if dst.device_type == DeviceType.QUANTIZE and len(self.data)<5:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[3])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], isMixtralWeight=isMixtralWeight)
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, isMixtralWeight=False, src_indices=None):
        if self.device == dst:
            return self, False
        

        return TorchTensor.create_from_torch(self.data.move_to(dst.name), dst), True
        # return self.copy(dst, isMixtralWeight=isMixtralWeight, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = TorchTensor.create_from_torch(self.data.move_to(dst.device), dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = name
        self.device_type = DeviceType.convert(self.dev)
        self.quantized_device = TorchHQQDevice(self)

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def allocate(self, shape, dtype, pin_memory=None, name=None, isMixtralWeight=False):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        # if not isinstance(dtype, ms.dtype):
        #     dtype = np_dtype_to_torch_dtype[dtype]
        if isMixtralWeight:
            dtype = ms.dtype.float16
        else:
            dtype = np_dtype_to_torch_dtype[dtype]
        data = ms.numpy.empty(shape, dtype=dtype)
        #data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = ops.concat((attention_mask.data,
             ops.ones((bs, 1), dtype=attention_mask.dtype)), axis=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def mixtral_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate):

        token_ids = inputs.data
        # token embedding
        token_embed = ops.embedding(token_ids, Parameter(w_token.data), None)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        return TorchTensor.create_from_torch(token_embed, self)
    
    def mixtral_output_embed(self, inputs, w_ln, w_token, donate, rms_norm_eps, do_sample, temperature):

        hidden = self.layerNorm(inputs.data, weight=w_ln.data, norm_eps = rms_norm_eps)

        if donate[0]: inputs.delete()

        logits = ops.dense(hidden, w_token.data)
        last_token_logits = logits.float()
        last_token_logits = logits[:,-1,:]
        dic={
            "hidden":Parameter(hidden),
            "w_token.data":Parameter(w_token.data),
            "last_token_logits":Parameter(last_token_logits),
        }

        if do_sample and not temperature < 1e-5:
            probs = ops.softmax(last_token_logits / temperature, axis=-1)
            ids = ops.multinomial(probs, 1, False)
        else:
            ids = last_token_logits.argmax(axis=1, keepdims=True)
        return TorchTensor.create_from_torch(ids, self)
    
    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_key_value_heads, int(config.hidden_size * (config.num_key_value_heads / config.num_attention_heads)), 
            task.prompt_len, task.gen_len, policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, config.torch_dtype, pin_memory=pin_memory)
        v_cache = self.allocate(shape, config.torch_dtype, pin_memory=pin_memory)
        return (k_cache, v_cache)
    
    def repeat_kv(self, hidden_states: Tensor, n_rep: int) -> Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return ops.cat((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def mixtral_mha(self, inputs, attention_mask, w_q, w_k, w_v, w_out, w_input_ln,
                    n_head, num_key_value_heads, donate, rotary_emb, sliding_window, sink_num, position_ids):
        """Multi-head attention (prefill phase)."""

        b, q_len, hidden_size = inputs.shape
        head_dim = hidden_size // n_head

        hidden = self.layerNorm(inputs.data, weight=w_input_ln.data)
        
        q = ops.dense(hidden, w_q.data)
        k = ops.dense(hidden, w_k.data)
        v = ops.dense(hidden, w_v.data)        

        q = q.view(b, q_len, n_head, head_dim).swapaxes(1, 2)
        k = k.view(b, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)
        v = v.view(b, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)

        kv_seq_len = attention_mask.data.shape[1]
        cos, sin = rotary_emb(v, seq_len=kv_seq_len)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        num_key_value_groups = n_head // num_key_value_heads
        k_repeat = self.repeat_kv(k, num_key_value_groups)
        v_repeat = self.repeat_kv(v, num_key_value_groups)

        idx = ops.arange(kv_seq_len)
        causal_mask = (idx <= idx.view(q_len, 1)).view(1, 1, q_len, q_len)
        
        mask = (attention_mask.data.view(b, 1, 1, q_len).to(ms.int32) & causal_mask.to(ms.int32)).to(ms.bool_)
        
        attn_weights = ops.matmul(q, k_repeat.swapaxes(2, 3)) / math.sqrt(head_dim)
        attn_weights = ops.where(mask, attn_weights, float(np.finfo(ms.dtype_to_nptype(q.dtype)).min))
        if attn_weights.shape != (b, n_head, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(b, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        attn_weights = ops.softmax(attn_weights, axis=-1,dtype=ms.float32).to(q.dtype)
        attn_output = ops.matmul(attn_weights, v_repeat)
        if attn_output.shape != (b, n_head, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(b, n_head, q_len, head_dim)}, but is"
                f" {attn_output.shape}"
            )
        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(b, q_len, hidden_size)

        attn_output = ops.dense(attn_output, w_out.data)
        attn_output= attn_output + inputs.data

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        k = k.reshape(b * num_key_value_heads, q_len, head_dim).permute(1, 0, 2)
        v = v.reshape(b * num_key_value_heads, q_len, head_dim).permute(1, 0, 2)
        
        k = TorchTensor.create_from_torch(k, self)
        v = TorchTensor.create_from_torch(v, self)
        return TorchTensor.create_from_torch(attn_output, self), (k,v)

    def mixtral_mha_gen(self, inputs, attention_mask, w_q, w_k, w_v, w_out, w_input_ln,
                     n_head, num_key_value_heads, donate, rotary_emb, sliding_window, sink_num, position_ids, k_cache, v_cache):
        """Multi-head attention (prefill phase)."""

        b, q_len, hidden_size = inputs.shape
        head_dim = hidden_size // n_head
        hidden = self.layerNorm(inputs.data, weight=w_input_ln.data)

        # shape: (b, s, h)
        q = ops.dense(hidden, w_q.data)
        k = ops.dense(hidden, w_k.data)
        v = ops.dense(hidden, w_v.data)
        q = q.view(b, q_len, n_head, head_dim).swapaxes(1, 2)
        k_new = k.view(b, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)
        v_new = v.view(b, q_len, num_key_value_heads, head_dim).swapaxes(1, 2)
        
        src_s = attention_mask.data.shape[1]
        cos, sin = rotary_emb(v_new, seq_len=src_s)
        q, k_new = self.apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)

        if isinstance(k_cache, TorchTensor):
            k_new = k_new.reshape( b * num_key_value_heads, 1,head_dim).permute(1,0,2)
            v_new = v_new.reshape( b * num_key_value_heads, 1,head_dim).permute(1,0,2)
            k = k_cache.data[:src_s]

            v = v_cache.data[:src_s]
            k[src_s - 1:src_s] = k_new
            v[src_s - 1:src_s] = v_new
            k = k.permute(1, 0, 2).reshape(b , num_key_value_heads, src_s, head_dim)
            # shape: (b * n_head, s, head_dim)
            v = v.permute(1, 0, 2).reshape(b , num_key_value_heads, src_s, head_dim)
        num_key_value_groups = n_head // num_key_value_heads
        k_repeat = self.repeat_kv(k, num_key_value_groups)
        v_repeat = self.repeat_kv(v, num_key_value_groups)
        mask = attention_mask.data.view(b, 1, 1, src_s)

        attn_weights = ops.matmul(q, k_repeat.swapaxes(2, 3)) / math.sqrt(head_dim)
        attn_weights = ops.where(mask, attn_weights, float(np.finfo(ms.dtype_to_nptype(q.dtype)).min))
        if attn_weights.shape != (b, n_head, q_len, src_s):
            raise ValueError(
                f"Attention weights should be of size {(b, n_head, q_len, src_s)}, but is"
                f" {attn_weights.shape}"
            )
        
        attn_weights = ops.softmax(attn_weights, axis=-1,dtype=ms.float32).to(q.dtype)
        attn_output = ops.matmul(attn_weights, v_repeat)
        if attn_output.shape!= (b, n_head, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(b, n_head, q_len, head_dim)}, but is"
                f" {attn_output.shape}"
            )
        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(b, q_len, hidden_size)
        attn_output = ops.dense(attn_output, w_out.data)
        attn_output=attn_output + inputs.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        k_new = TorchTensor.create_from_torch(k_new, self)
        v_new = TorchTensor.create_from_torch(v_new, self)
        return TorchTensor.create_from_torch(attn_output, self), (k_new, v_new)
    
    def mixtral_gate(self, inputs, w_gate, w_ffn_ln, config, donate):
        out = self.layerNorm(inputs.data, weight=w_ffn_ln.data)
        global countttgate

        countttgate+=1
        
        _, _, hidden_dim = out.shape
        data = out
        hidden = data.view(-1, hidden_dim)
        router_logits = ops.dense(hidden, w_gate.data)

        routing_weights = ops.softmax(router_logits, axis=1,dtype=ms.float32)
        routing_weights, selected_experts = ops.topk(routing_weights, config.num_experts_per_tok, dim=-1)
        routing_weights /= routing_weights.sum(axis=-1, keepdims =True)
        # cast back to the input dtype
        routing_weights = routing_weights.to(out.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = mint.nn.functional.one_hot(selected_experts, num_classes=config.num_local_experts).permute(2, 1, 0)

        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self), expert_mask, routing_weights, selected_experts
    
    def sparseMLP(self, inputs, w1, w2, w3, routing_weights):
        out = ops.silu(ops.dense(inputs, w1.data)) * ops.dense(inputs, w3.data)
        out = ops.dense(out, w2.data)
        out = routing_weights * out

        return out
        # return TorchTensor.create_from_torch(out, self)

    def mixtral_MoE(self, inputs, residual, experts, hot_experts, prefetch_experts, load_experts, gate_bufs, transfer_expert_list, num_gpu_batches):
        # hot(prefetch) experts computation
        batch_size, sequence_length, hidden_dim = inputs[0].val.shape
        # result = array_1d(num_gpu_batches, ValueHolder)
        results = {}
        for k in range(num_gpu_batches):
            results[k] = ms.ops.zeros((batch_size * sequence_length, hidden_dim), dtype=inputs[0].val.dtype)
            # results[k] = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=inputs[0].val.dtype, device=inputs[0].val.device.dev)
        for index in hot_experts:
            for batch, (input, gate_buf) in enumerate(zip(inputs, gate_bufs)):
                input = input.val
                input.data = input.data.view(-1, hidden_dim)
                expert_mask, routing_weights, _ = gate_buf.val
                idx, top_x = ops.nonzero(expert_mask[index],as_tuple=True)
                if top_x.shape[0] == 0:
                    continue
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                current_state = input.data[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = experts[f"expert_{index}"].forward(current_state, prefetch_experts[f"expert_{index}"], routing_weights[top_x_list, idx_list, None])

                results[batch].index_add(0, top_x.to(ms.int32), current_hidden_states.to(input.dtype))
            transfer_expert_list.remove(index)
        hot_experts.clear()

        # load experts computation
        load_experts = load_experts.pop()
        while transfer_expert_list:
            expert_idx = transfer_expert_list.pop(0)
            # print(expert_idx)
            # if load_experts[f"expert_{index}"].is_cuda:
            for batch, (input, gate_buf) in enumerate(zip(inputs, gate_bufs)):
                input = input.val
                input.data = input.data.view(-1, hidden_dim)
                expert_mask, routing_weights, _ = gate_buf.val
                idx, top_x = ops.nonzero(expert_mask[expert_idx],as_tuple=True)
                if top_x.shape[0] == 0:
                    continue
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                current_state = input.data[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = experts[f"expert_{expert_idx}"].forward(current_state, load_experts[f"expert_{expert_idx}"], routing_weights[top_x_list, idx_list, None])

                results[batch].index_add(0, top_x.to(ms.int32), current_hidden_states.to(input.dtype))

        if len(transfer_expert_list) == 0:
            for k in range(num_gpu_batches):
                results[k] = results[k].reshape(batch_size, sequence_length, hidden_dim)
                # print(residual[k].val)
                results[k]=results[k] +residual[k].val.data
                results[k] = TorchTensor.create_from_torch(results[k], self)
        else:
            print("didn't finish computing.")
            assert False

        # if donate[0]: inputs.delete()
        return results
        # return TorchTensor.create_from_torch(final_hidden_states, self)
    
    def MoE_computation(self, inputs, experts, prefetch_experts, gate_buf, donate):
        # dequantize weights
        batch_size, sequence_length, hidden_dim = inputs.shape
        inputs.data = inputs.data.view(-1, hidden_dim)
        
        #final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=inputs.dtype, device=inputs.device.dev)
        final_hidden_states = ops.zeros((batch_size * sequence_length, hidden_dim), dtype=inputs.dtype)

        (expert_mask, routing_weights, _) = gate_buf.val

        for expert_idx, expert in enumerate(experts.values()):
            idx, top_x = ops.nonzero(expert_mask[expert_idx],as_tuple=True)
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = inputs.data[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert.forward(current_state, prefetch_experts[f"expert_{expert_idx}"], routing_weights[top_x_list, idx_list, None])
            final_hidden_states.index_add(0, top_x.to(ms.int32), current_hidden_states.to(inputs.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(final_hidden_states, self)

    def layerNorm(self, inputs, weight, norm_eps = 1e-05):
        #mixtral norm_eps=le-05
        variance = inputs.to(ms.dtype.float32).pow(2).mean(-1, keep_dims=True)
        variance_epsilon = norm_eps
        inputs = inputs * ops.rsqrt(variance + variance_epsilon)

        # convert into half-precision if necessary
        if weight.dtype in [ms.dtype.float16, ms.dtype.bfloat16]:
            inputs = inputs.to(weight.dtype)

        return weight * inputs

    def synchronize(self):
        ms.hal.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = ms.hal.memory_allocated(self.dev)
            peak_mem = ms.hal.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        elif self.device_type == DeviceType.Ascend:
            cur_mem = ms.hal.memory_allocated(self.dev)
            peak_mem = ms.hal.max_memory_allocated(self.dev)
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        ms.hal.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, device_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.quantized_device = TorchHQQDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, device_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def allocate(self, shape, dtype, pin_memory=None, name=None, isMixtralWeight=False):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype], path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_key_value_heads, int(config.hidden_size * (config.num_key_value_heads / config.num_attention_heads)), 
            task.prompt_len, task.gen_len, policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float32, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float32, pin_memory=pin_memory)
        return (k_cache, v_cache)

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()

def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if (src.device.device_type == DeviceType.QUANTIZE or
          dst.device.device_type == DeviceType.QUANTIZE):
          
        # The tensor is quantized, do recursive calls
        print("False ")
        general_copy(dst.data[0], None, src.data[0], None)
        general_copy(dst.data[1], None, src.data[1], None)
        general_copy(dst.data[2], None, src.data[2], None)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        print("False ")

        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        print("False ")

        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        print("False ")
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        print("False ")
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)
    elif (src.device.device_type == DeviceType.Ascend and
          dst.device.device_type == DeviceType.CPU):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        if dst_indices:
            update = ops.InplaceUpdateV2()
            update=update.set_device("CPU")
            indexs = tuple(range(dst_indices[0].start,dst_indices[0].stop))
            dst.data= update(dst.data,indexs,src.move_to("CPU"))
        else:
            dst.data=src.move_to("CPU")

    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data

        if dst_indices:
            dst.data[dst_indices] = src
        else:
            dst.data=src

def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = Tensor.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    if ops.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data

def copy_worker_func(queue, device_id):
    """The copy worker thread."""
    # ms.context.set_context(device_target="Ascend", device_id=device_id)
    
    #cpu
    cpu_buf = ms.Tensor(np.empty((3, 2)))
    copy_stream = ms.hal.Stream()

    with ms.hal.StreamCtx(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return
            
            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if ((src.device.device_type == DeviceType.CUDA) or
                (dst.device.device_type == DeviceType.CUDA)):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
