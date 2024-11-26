import logging
import os
from typing import Callable, Dict, Optional
import difflib
from copy import deepcopy

from mindspore import load_checkpoint, load_param_into_net, nn

from pgnet.models.download import DownLoad

__all__ = ["load_model", "drop_inconsistent_shape_parameters", "set_amp_attr"]
_logger = logging.getLogger(__name__)

_DEFAULT_DOWNLOAD_ROOT = os.path.join(os.path.expanduser("~"), ".mindspore")

def drop_inconsistent_shape_parameters(model, param_dict):
    updated_param_dict = dict()
    for param in model.get_parameters():
        name = param.name
        if name in param_dict:
            if param_dict[name].shape == param.shape:
                updated_param_dict[name] = param_dict[name]
            else:
                _logger.warning(
                    f"Dropping checkpoint parameter `{name}` with shape `{param_dict[name].shape}`, "
                    f"which is inconsistent with cell shape `{param.shape}`"
                )
        else:
            _logger.warning(f"Cannot find checkpoint parameter `{name}`.")
    return updated_param_dict

def auto_map(model, param_dict):
    """Raname part of the param_dict such that names from checkpoint and model are consistent"""
    updated_param_dict = deepcopy(param_dict)
    net_param = model.get_parameters()
    ckpt_param = list(updated_param_dict.keys())
    remap = {}
    for param in net_param:
        if param.name not in ckpt_param:
            _logger.info(f'Cannot find a param to load: {param.name}')
            poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
            if len(poss) > 0:
                _logger.info(f'=> Find most matched param: {poss[0]},  loaded')
                updated_param_dict[param.name] = updated_param_dict.pop(poss[0])  # replace
                remap[param.name] = poss[0]
            else:
                raise ValueError('Cannot find any matching param from: ', ckpt_param)

    if remap != {}:
        _logger.warning('Auto mapping succeed. Please check the found mapping names to ensure correctness')
        _logger.info('\tNet Param\t<---\tCkpt Param')
        for k in remap:
            _logger.info(f'\t{k}\t<---\t{remap[k]}')
    return updated_param_dict


def get_default_download_root():
    return deepcopy(_DEFAULT_DOWNLOAD_ROOT)

def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")

def download_pretrained(default_cfg):
    """Download the pretrained ckpt from url to local path"""
    if "url" not in default_cfg or not default_cfg["url"]:
        _logger.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    file_path = DownLoad().download_url(default_cfg["url"], path=download_path)
    return file_path

def load_model(
    network,
    load_from: Optional[str] = None,
    filter_fn: Optional[Callable[[Dict], Dict]] = None,
    auto_mapping: bool = False,
    strict: bool = False,
):
    if load_from is None:
        return

    if load_from[:4] == "http":
        url_cfg = {"url": load_from}
        local_ckpt_path = download_pretrained(url_cfg)
    else:
        local_ckpt_path = load_from

    assert local_ckpt_path and os.path.exists(local_ckpt_path), (
        f"Failed to load checkpoint. `{local_ckpt_path}` NOT exist. \n"
        "Please check the path and set it in `eval-ckpt_load_path` or `model-pretrained` in the yaml config file "
    )

    params = load_checkpoint(local_ckpt_path)

    if filter_fn is not None:
        params = filter_fn(params)

    if auto_mapping:
        params = auto_map(network, params)

    if not strict:
        params = drop_inconsistent_shape_parameters(network, params)

    load_param_into_net(network, params, strict_load=strict)

    _logger.info(
        f"Finish loading model checkoint from {load_from}. "
        "If no parameter fail-load warning displayed, all checkpoint params have been successfully loaded."
    )


def set_amp_attr(network : nn.Cell, amp_level : str):
    cells = network.name_cells()
    for name in cells:
        setattr(network._cells[name], "_amp_level", amp_level)
