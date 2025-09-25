import argparse
import glob
import os
from quantizer import quantize
from tqdm import tqdm
import re

def download_Qwen_weights(model_name, path):
    # from safetensors.torch import load_file, save_file
    from mindnlp.core.serialization import safe_load_file as load_file
    from mindnlp.core.serialization import safe_save_file as save_file

    from huggingface_hub import snapshot_download
    folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    # weights_path=f'model_weights/{model_name}/weight'
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))
    # print("weights_path",weights_path)
    # print("safetensor_files",safetensor_files)

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    ori_path = os.path.join(path, 'original_new')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    expert_files = {}
    expert_int4_files = {}
    expert_int2_files = {}
    for layer in range(24):
        expert_files[layer] = {}
        expert_int4_files[layer] = {}
        expert_int2_files[layer] = {}
        for expert in range(60):
            expert_files[layer][expert] = {}
            expert_int4_files[layer][expert] = {}
            expert_int2_files[layer][expert] = {}

    expert_pattern = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight")

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)

        for name, param in tqdm(state.items(), leave=False):
            if "shared" in name or "expert" not in name:
                param_path = os.path.join(ori_path, name)
                save_file({"tensor": param}, param_path)
            else:
                match = expert_pattern.search(name)
                layer, expert_index, proj_type = match.groups()
                layer = int(layer)
                expert_index = int(expert_index)
                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)
                expert_int4_files[layer][expert_index][f'{proj_type}_nbits'] = param_int4.pop('nbits')
                expert_int4_files[layer][expert_index][f'{proj_type}_shape'] = param_int4.pop('shape')
                expert_int4_files[layer][expert_index][f'{proj_type}'] = param_int4.pop('W_q')
                expert_int4_files[layer][expert_index][f'{proj_type}_scale'] = param_int4.pop('scale')
                expert_int4_files[layer][expert_index][f'{proj_type}_zero'] = param_int4.pop('zero')

                expert_int2_files[layer][expert_index][f'{proj_type}_nbits'] = param_int2.pop('nbits')
                expert_int2_files[layer][expert_index][f'{proj_type}_shape'] = param_int2.pop('shape')
                expert_int2_files[layer][expert_index][f'{proj_type}'] = param_int2.pop('W_q')
                expert_int2_files[layer][expert_index][f'{proj_type}_scale'] = param_int2.pop('scale')
                expert_int2_files[layer][expert_index][f'{proj_type}_zero'] = param_int2.pop('zero')
    
    # for layer_id, experts in expert_files.items():
    #     for expert_id, expert_data in experts.items():
    #         expert_path = os.path.join(ori_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
    #         save_file(expert_data, expert_path)
    for layer_id, experts in expert_int4_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int4_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)
    for layer_id, experts in expert_int2_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int2_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="/data1/hyg/Fate/model_weights")
    args = parser.parse_args()
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE)

    model = "Qwen/Qwen1.5-MoE-A2.7B"
    download_Qwen_weights(model, args.path)