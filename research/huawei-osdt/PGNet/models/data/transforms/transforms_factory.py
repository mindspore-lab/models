import logging
from typing import Dict, List

import numpy as np

from .pgnet_transforms import *
from .e2e_transforms import *
from .general_transforms import *
from .det_transforms import *

__all__ = ["create_transforms", "run_transforms"]
_logger = logging.getLogger(__name__)

def create_transforms(transform_pipeline: List, global_config: Dict = None):
    assert isinstance(
        transform_pipeline, list
    ), f"transform_pipeline config should be a list, but {type(transform_pipeline)} detected"

    transforms = []
    for transform_config in transform_pipeline:
        if isinstance(transform_config, dict):
            assert len(transform_config) == 1, "yaml format error in transforms"
            trans_name = list(transform_config.keys())[0]
            param = {} if transform_config[trans_name] is None else transform_config[trans_name]
            if global_config is not None:
                param.update(global_config)

            transform = eval(trans_name)(**param)
            transforms.append(transform)
        elif callable(transform_config):
            transforms.append(transform_config)
        else:
            raise TypeError("transform_config must be a dict or a callable instance")

    return transforms

def run_transforms(data, transforms=None, verbose=False):
    if transforms is None:
        transforms = []
    for i, transform in enumerate(transforms):
        if verbose:
            _logger.info(f"Trans {i}: {transform}")
            _logger.info(
                "\tInput: " + "\t".join([f"{k}: {data[k].shape}" for k in data if isinstance(data[k], np.ndarray)])
            )
        data = transform(data)
        if verbose:
            _logger.info(
                "\tOutput: " + "\t".join([f"{k}: {data[k].shape}" for k in data if isinstance(data[k], np.ndarray)])
            )

        if data is None:
            raise RuntimeError(f"Empty result is returned from transform `{transform}`")
    return data
