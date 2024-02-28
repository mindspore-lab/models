import torch
import mindspore
from mindspore import Tensor

def convert_hf_to_ms_clip(pretrained_path, write_path=None):
    
    ms_ckpt = []
    hf_model_dict = torch.load(f'{pretrained_path}/pytorch_model.bin')
    for k, v in hf_model_dict.items():
        if 'position_ids' in k: continue
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
        if 'layer_norm' in k or 'pre_layrnorm' in k or 'post_layernorm' in k:
            if '.weight' in k:
                k = k.replace('.weight', '.gamma')
            if '.bias' in k:
                k = k.replace('.bias', '.beta')
        if 'embeddings' in k and 'patch_embedding' not in k:
            k = k.replace('weight', 'embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})
        
    if write_path:
        mindspore.save_checkpoint(ms_ckpt, write_path)
    return ms_ckpt

def verify_net(pretrained_path):
    from mindnlp.models import BertConfig, CLIPVisionConfig, CLIPTextConfig, BertModel, CLIPModel, CLIPConfig
    from mindspore import load_checkpoint, load_param_into_net
    
    clip_config = CLIPConfig.from_pretrained(pretrained_path)
    clip_config.text_config = CLIPTextConfig(**clip_config.text_config)
    clip_config.vision_config = CLIPVisionConfig(**clip_config.vision_config)
    ms_model = CLIPModel(clip_config)
    clip_ckpt = mindspore.load_checkpoint('clip-vit-base-patch32/mindspore.ckpt')
    load_param_into_net(ms_model, clip_ckpt)


convert_hf_to_ms_clip('clip-vit-base-patch32', 'clip-vit-base-patch32/mindspore.ckpt')
verify_net('clip-vit-base-patch32/')