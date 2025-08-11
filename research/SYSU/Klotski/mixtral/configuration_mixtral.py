import glob
import os
import torch

import numpy as np
from tqdm import tqdm
from ml_dtypes import bfloat16
import mindspore as ms
import mindspore.context as context
import psutil
from utils import MB

class MixtralConfig:
    name: str = "mixtral-8x7b-v0.1"
    # keys_to_ignore_at_inference: list = ["past_key_values"]
    vocab_size: int = 32000
    hidden_size: int = 4096
    input_dim: int = 4096
    intermediate_size: int = 14336
    max_position_embeddings: int =  32768
    num_hidden_layers: int = 10  #32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    # max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    pad_token_id: int = 2
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: int = 1e6
    # sliding_window: int = 4096
    sliding_window = None
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.02
    dtype: type = np.float16
    #有问题
    torch_dtype: type = ms.dtype.float16

    def model_bytes(self):
        h = self.hidden_size
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        (h * h + h * h / 4) * 2 +
        # moe(expert + gate)
        (h * self.intermediate_size) * 3 * 8 + 8 * h +
        # layer norm
        h * 2) +
        # embedding * 2 + final_ln
        self.vocab_size * h * 2 + h)

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.hidden_size * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.hidden_size * 2

    def memory_cost_mixtral(self, batch_size, seq_len, device, quantize, GPU_memory_budget=0):
        
        #初始化backen
        a=ms.Tensor([1,1])
        b=ms.Tensor([1,1])
        print(a+b)


        if GPU_memory_budget == 0:
            total_memory = ms.hal.get_device_properties(device).total_memory / (1024**2)
            # allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
            # gpu_info = context.get_gpu_status()
            allocated_memory = ms.hal.memory_stats()["total_allocatd_memory"]     
            GPU_memory_budget = total_memory - allocated_memory
        else:
            GPU_memory_budget = GPU_memory_budget * 1024

        # CPU
        CPU_memory_budget = psutil.virtual_memory().available / (1024 * 1024)

        # Reserve space for 1024
        # seq_len = 1024
        num_hidden_layers = self.num_hidden_layers
        hidden_size = self.hidden_size
        vocab_size = self.vocab_size
        moe_intermediate_size = self.intermediate_size
        num_experts = self.num_local_experts

        num_heads = self.num_attention_heads
        head_dim = hidden_size // num_heads
        num_key_value_heads = self.num_key_value_heads
        max_position_embeddings = self.max_position_embeddings

        # layer
        embed = (vocab_size * hidden_size * 2 * 2)/MB
        attention = (2 * (hidden_size * num_heads * head_dim + 
                            hidden_size * num_key_value_heads * head_dim) 
                            * num_hidden_layers * 2)/MB
        rotaryEmbedding = ((head_dim // 2 + head_dim * max_position_embeddings * 2) * num_hidden_layers * 4)/MB
        norm = ((2 * hidden_size * num_hidden_layers + hidden_size) * 2)/MB

        expert = (3 * hidden_size * moe_intermediate_size * 2)/MB
        expert_gate = (hidden_size * num_experts * num_hidden_layers * 2)/MB
        
        # during infer.
        kv = (2 * batch_size * seq_len * hidden_size * 2)/MB
        all_kv = kv * num_hidden_layers
        hidden = (batch_size * seq_len * hidden_size * 2)/MB

        # remove meta_data, prefetch space, keep space for inference
        GPU_available_memory = GPU_memory_budget - (expert * 8) - 2000

        dense_part = embed + attention + rotaryEmbedding + norm + expert_gate + hidden

        if quantize:
            zero_scale = (2 * (moe_intermediate_size // 64 * hidden_size) * 1 * 4)/MB
            expert = expert // 4 + 3 * zero_scale
        expert_layer = 8 * expert

        expert_offload_map = {}
        kv_offload_map = {}
        for i in range(num_hidden_layers):
            expert_offload_map[i] = 1
            kv_offload_map[i] = 1

        if GPU_available_memory < dense_part:
            if CPU_memory_budget < dense_part:
                return (2, 2, 2)
            else:
                CPU_memory_budget = CPU_memory_budget - dense_part
                if CPU_memory_budget < all_kv:
                    m = int(CPU_memory_budget // kv)
                    for i in range(m, num_hidden_layers, 1):
                        kv_offload_map[i] = 2
                    return (2, 1, kv_offload_map)
                else:
                    CPU_memory_budget = CPU_memory_budget - all_kv
                    if CPU_memory_budget > num_hidden_layers * expert_layer:
                        return (1, 1, 1)
                    else:
                        n = int(CPU_memory_budget // expert_layer)
                        for i in range(n, num_hidden_layers, 1):
                            expert_offload_map[i] = 2
                        return (expert_offload_map, 1, 1)
        else:
            GPU_available_memory = GPU_available_memory - dense_part
            if GPU_available_memory < all_kv:
                m = int(GPU_available_memory // kv)
                for i in range(m):
                    kv_offload_map[i] = 0
                rest_kv = GPU_available_memory - m * kv
                if CPU_memory_budget < rest_kv:
                    mm = int(CPU_memory_budget // kv)
                    for i in range(m + mm, num_hidden_layers, 1):
                        kv_offload_map[i] = 2
                    return (2, 0, kv_offload_map)
                else:
                    CPU_memory_budget = CPU_memory_budget - rest_kv
                    if CPU_memory_budget > num_hidden_layers * expert_layer:
                        return (1, 0, kv_offload_map)
                    else:
                        n = int(CPU_memory_budget // expert_layer)
                        for i in range(n, num_hidden_layers, 1):
                            expert_offload_map[i] = 2
                        return (expert_offload_map, 0, kv_offload_map)
            else:
                GPU_available_memory = GPU_available_memory - all_kv
                if GPU_available_memory > num_hidden_layers * expert_layer:
                    return (0, 0, 0)
                else:
                    n = int(GPU_available_memory // expert_layer)
                    for i in range(0, n, 1):
                        expert_offload_map[i] = 0
                    if CPU_memory_budget > (num_hidden_layers - n) * expert_layer:
                        return (expert_offload_map, 0, 0)
                    else:
                        nn = int(CPU_memory_budget // expert_layer)
                        for i in range(n + nn, num_hidden_layers, 1):
                            expert_offload_map[i] = 2
                        return (expert_offload_map, 0, 0)

def get_mixtral_config(name):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    if name == "mixtral-8x7b-v0.1":
        config = MixtralConfig()
    elif name == "mixtral-8x22b-v0.1":
        config = MixtralConfig()
        config.name = name
        config.hidden_size = 6144
        config.intermediate_size = 16384
        config.max_position_embeddings = 65536
        config.num_attention_heads = 48
        config.num_hidden_layers = 56
    else:
        raise ValueError(f"Invalid model name: {name}")

    return config

def download_mixtral_weights(model_name, path):
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    # from mindnlp.core.serialization import safe_load_file
    # load from HF
    hf_model_name = "mistralai/" + model_name
    folder = snapshot_download(hf_model_name, allow_patterns="*.safetensors")
    safetensors_files = glob.glob(os.path.join(folder, "*.safetensors"))

    # load from local
    # hf_model_name = "/root/workspace/mixtral-8x7b"
    # safetensors_files = glob.glob(os.path.join(hf_model_name, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for safetensors_file in tqdm(safetensors_files, desc="Convert format"):
        state = load_file(safetensors_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            param_fp32 = param.float()
            arr_bf16 = param_fp32.cpu().detach().numpy()
            # arr_bf16=param.asnumpy()
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, arr_bf16)

# download_mixtral_weights from *.pt
def download_mixtral_weights_from_pt(model_name, path):
    # from huggingface_hub import snapshot_download

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"and convert them.")

    # hf_model_name = "mistralai/" + model_name
    # folder = snapshot_download(hf_model_name, allow_patterns="*.pt")
    # pt_files = glob.glob(os.path.join(folder, "*.pt"))

    # load from local
    if model_name == "mixtral-8x7b-v0.1":
        hf_model_name = "/home/ma-user/work/mixtral"
    elif model_name == "mixtral-8x22b-v0.1":
        hf_model_name = "/home/ma-user/work/mixtral"
    safetensors_files = glob.glob(os.path.join(hf_model_name, "*.pt"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for safetensors_file in tqdm(safetensors_files, desc="Convert format"):
        state = ms.load(safetensors_file)
        for name, param in tqdm(state.items(), leave=False):
            # txt_file.write(f"{name}: {111}\n")                
            if "block_sparse_moe.w" in name :
                num_chunks = 8
                chunks = ms.ops.chunk(param, chunks=num_chunks, axis=0)
                for i, chunk in enumerate(chunks):
                    if "block_sparse_moe.w2" in name :
                        chunk = chunk.T
                    param_path = os.path.join(path, name)
                    name_parts = param_path.split(".")[-1]
                    new_param_path = param_path.replace(name_parts, f"expert{i}.{name_parts}")
                    param_float32 = chunk.float()
                    with open(new_param_path, "wb") as f:
                        np.save(f, param_float32.cpu().detach().numpy())
            else:
                param_path = os.path.join(path, name)
                param_float32 = param.float()
                with open(param_path, "wb") as f:
                    np.save(f, param_float32.cpu().detach().numpy())


