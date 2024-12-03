import os.path

import mindspore as ms
from mindformers import AutoModel, Blip2Classifier
from mindspore import nn

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .utils import logger
from .utils.utils import freeze_layer


def build_sam_vit_h(checkpoint=None, enable_text_encoder=False, text_encoder_config=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        enable_text_encoder=enable_text_encoder,
        text_encoder_config=text_encoder_config
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, enable_text_encoder=False, text_encoder_config=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        enable_text_encoder=enable_text_encoder,
        text_encoder_config=text_encoder_config
    )


def build_sam_vit_b(checkpoint=None, enable_text_encoder=False, text_encoder_config=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        enable_text_encoder=enable_text_encoder,
        text_encoder_config=text_encoder_config,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    enable_text_encoder=False,
    text_encoder_config=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # blip2 model default to load from ./checkpoint_download/blip2/blip2_stage1_classification.ckpt
    if text_encoder_config is None:
        text_encoder_config = dict()
    text_encoder: Blip2Classifier = \
        AutoModel.from_pretrained(text_encoder_config.get('type', 'blip2_stage1_classification')) \
            if enable_text_encoder else None
    text_feature_dim = text_encoder_config.get('feature_dim', prompt_embed_dim)

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            # use approximate=False to be close to pytorch, ref:
            # https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/GELU.html?highlight=gelu
            act_layer=partial(nn.GELU, approximate=False),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            text_feature_dim=text_feature_dim,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        text_encoder=text_encoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.set_train(False)
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint} does not exist')
        print(f'load checkpoint from {checkpoint}')
        ms.load_checkpoint(checkpoint, sam)
    return sam


def create_model(args):
    model = sam_model_registry[args.type](checkpoint=args.checkpoint,
                                          enable_text_encoder=args.get('enable_text_encoder', False),
                                          text_encoder_config=args.get('text_encoder', None))
    if args.freeze is not None:
        for module in ['image_encoder', 'prompt_encoder', 'mask_decoder', 'text_encoder']:
            if not args.freeze.get(module, False):
                continue
            filter_prefix = getattr(args.freeze.get(module), 'filter_prefix', None)
            specify_prefix = getattr(args.freeze.get(module), 'specify_prefix', None)
            logger.info(f'freezing {module}, filter_prefix: {filter_prefix}, specify_prefix: {specify_prefix}')
            freeze_layer(getattr(model, module), filter_prefix=filter_prefix, specify_prefix=specify_prefix)
    return model
