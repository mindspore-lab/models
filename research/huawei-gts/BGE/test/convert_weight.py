from mindspore import load_checkpoint
import torch
ckpt = load_checkpoint("./path/to/ckpt")

torch_bin = {}
for key, value in ckpt.items():
    if "adam_m" not in key and "adam_v" not in key:
        try:
            torch_bin[key.replace("model.", "")] = torch.Tensor(value.numpy())
        except ValueError:
            print(key)

torch.save(torch_bin, "./path/to/pytorch_model.bin")