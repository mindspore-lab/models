from typing import Union

from mindspore.amp import auto_mixed_precision

from ._registry import is_model, list_models, model_entrypoint
from .base_model import BaseModel
from models.networks.load_model import load_model, set_amp_attr

__all__ = ["build_model"]


def build_model(name_or_config: Union[str, dict], **kwargs):
    is_customized_model = True
    if isinstance(name_or_config, str):
        # build model by specific model name
        model_name = name_or_config
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            network = create_fn(**kwargs)
        else:
            raise ValueError(
                f"Invalid model name: {model_name}. Supported models are {list_models()}"
            )
        is_customized_model = False
    elif isinstance(name_or_config, dict):
        network = BaseModel(name_or_config)
    else:
        raise ValueError("Type error for config")

    # load checkpoint
    if "ckpt_load_path" in kwargs:
        load_from = kwargs["ckpt_load_path"]
        if isinstance(load_from, bool) and is_customized_model:
            raise ValueError(
                "Cannot find the pretrained checkpoint for a customized model without giving the url or local path "
                "to the checkpoint.\nPlease specify the url or local path by setting `model-pretrained` (if training) "
                "or `eval-ckpt_load_path` (if evaluation) in the yaml config"
            )

        load_model(network, load_from)

    if "amp_level" in kwargs:
        auto_mixed_precision(network, amp_level=kwargs["amp_level"])
        set_amp_attr(network, kwargs["amp_level"])

    return network
