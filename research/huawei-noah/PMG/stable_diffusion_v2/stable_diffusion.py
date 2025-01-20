"""
Text to image generation
"""
import argparse
import logging
import os
import sys
import time

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, workspace)
import numpy as np
from common import init_env
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.uni_pc import UniPCSampler
from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from ldm.util import instantiate_from_config, str2bool
from tools.safety_checker import SafetyChecker
from tools.watermark import WatermarkEmbedder
from utils import model_utils
from utils.download import download_checkpoint
from utils.long_prompt import get_text_embeddings

logger = logging.getLogger("text_to_image")
_version_cfg = {
    "2.1": ("sd_v2-1_base-7c8d09ce.ckpt", "v2-inference.yaml", 512),
    "2.1-v": ("sd_v2-1_768_v-061732d1.ckpt", "v2-vpred-inference.yaml", 768),
    "2.0": ("sd_v2_base-57526ee4.ckpt", "v2-inference.yaml", 512),
    "2.0-v": ("sd_v2_768_v-e12e3a9b.ckpt", "v2-vpred-inference.yaml", 768),
    "1.5": ("sd_v1.5-d0ab7146.ckpt", "v1-inference.yaml", 512),
    "1.5-wukong": ("wukong-huahua-ms.ckpt", "v1-inference-chinese.yaml", 512),
}
_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"
CLIP_CKPT_URL = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt"
_MIN_CKPT_SIZE = 4.0 * 1e9

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(
    config, ckpt, use_lora=False, lora_rank=4, lora_fp16=True, lora_only_ckpt=None, ti_only_ckpt=None
):
    model = instantiate_from_config(config.model)

    def _load_model(_model, ckpt_fp, verbose=True, filter=None):
        if os.path.exists(ckpt_fp):
            param_dict = ms.load_checkpoint(ckpt_fp)
            if param_dict:
                param_not_load, ckpt_not_load = model_utils.load_param_into_net_with_filter(
                    _model, param_dict, filter=filter
                )
                if verbose:
                    if len(param_not_load) > 0:
                        logger.info(
                            "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                        )
        else:
            logger.error(f"!!!Error!!!: {ckpt_fp} doesn't exist")
            raise FileNotFoundError(f"{ckpt_fp} doesn't exist")

    if use_lora:
        load_lora_only = True if lora_only_ckpt is not None else False
        if not load_lora_only:
            logger.info(f"Loading model from {ckpt}")
            _load_model(model, ckpt)
        else:
            if os.path.exists(lora_only_ckpt):
                lora_param_dict = ms.load_checkpoint(lora_only_ckpt)
                if "lora_rank" in lora_param_dict.keys():
                    lora_rank = int(lora_param_dict["lora_rank"].value())
                    logger.info(f"Lora rank is set to {lora_rank} according to the found value in lora checkpoint.")
            else:
                raise ValueError(f"{lora_only_ckpt} doesn't exist")
            # load the main pretrained model
            logger.info(f"Loading pretrained model from {ckpt}")
            _load_model(model, ckpt, verbose=True, filter=ms.load_checkpoint(ckpt).keys())
            injected_attns, injected_trainable_params = inject_trainable_lora(
                model,
                rank=lora_rank,
                use_fp16=(model.model.diffusion_model.dtype == ms.float16),
                scale=1.0,
            )

            # load fine-tuned lora params
            logger.info(f"Loading LoRA params from {lora_only_ckpt}")
            _load_model(model, lora_only_ckpt, verbose=True, filter=injected_trainable_params.keys())
    else:
        logger.info(f"Loading model from {ckpt}")
        _load_model(model, ckpt)
    if ti_only_ckpt is not None:
        from ldm.modules.textual_inversion.manager import TextualInversionManager

        logger.info(f"Loading Textual Inversion params from {ti_only_ckpt}")
        manager = TextualInversionManager(
            model,
        )
        manager.load_checkpoint_textual_inversion(ti_only_ckpt)

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False
    if ti_only_ckpt is not None:
        return model, manager
    return model

class SDPipeline():
    def __init__(self, version='2.1-v'):
        super().__init__()
        self.version = version
        ckpt_name = _version_cfg[version][0]
        self.ckpt_path = os.path.join(workspace, "models/", ckpt_name)
        self.size = _version_cfg[version][2]
        self.H = self.size
        self.W = self.size
        self.config = os.path.join("configs", _version_cfg[version][1])
        self.scale = 9.0
        self.ms_mode = 0
        self.use_parallel = False
        self.fixed_code = True
        self.device_target = 'Ascend'
        self.seed = 42
        self.ddim_eta = 0
                
        # init
        device_id, self.rank_id, self.device_num = init_env(
            self.ms_mode,
            seed=self.seed,
            distributed=self.use_parallel,
            device_target=self.device_target,
        )

        work_dir = os.path.dirname(os.path.abspath(__file__))


        # create model
        if not os.path.isabs(self.config):
            self.config = os.path.join(work_dir, self.config)
        config = OmegaConf.load(f"{self.config}")
        self.model = load_model_from_config(
            config,
            ckpt=self.ckpt_path,
        )

        self.prediction_type = getattr(config.model, "prediction_type", "noise")
        logger.info(f"Prediction type: {self.prediction_type}")
        

            
            
    def generate(self, prompt="", negative_prompt="", n_samples=1, n_iter=1, sampling_steps=20, sampler_type = "dpm_solver"):
        batch_size = n_samples
        
        if sampler_type == 'dpm_solver':
            self.sampler = DPMSolverSampler(self.model, "dpmsolver", prediction_type=self.prediction_type)
        elif sampler_type == 'ddim':
            self.sampler = DDIMSampler(self.model)
        else:
            self.sampler = DPMSolverSampler(self.model, "dpmsolver++", prediction_type=self.prediction_type)
            
        # infer
        start_code = None
        if self.fixed_code:
            stdnormal = ms.ops.StandardNormal()
            start_code = stdnormal((n_samples, 4, self.H // 8, self.W // 8))

        gen_images = []

        for n in range(n_iter):
            uc = None
            if self.scale != 1.0:
                if isinstance(negative_prompt, tuple):
                    negative_prompt = list(negative_prompt)
            else:
                negative_prompt = None
            if isinstance(prompt, tuple):
                prompt = list(prompt)
            c, uc = get_text_embeddings(
                self.model, prompt, negative_prompt
            )
            shape = [4, self.H // 8, self.W // 8]
            samples_ddim, _ = self.sampler.sample(
                S=sampling_steps,
                conditioning=c,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=self.scale,
                unconditional_conditioning=uc,
                eta=self.ddim_eta,
                x_T=start_code,
            )
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            for x_sample in x_samples_ddim_numpy:
                x_sample = 255.0 * x_sample.transpose(1, 2, 0)
                img = Image.fromarray(x_sample.astype(np.uint8))
                gen_images.append(img)
                
        return gen_images