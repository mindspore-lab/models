"""
Create and run transformations from a config or predefined transformation pipeline
"""
import logging
from typing import Dict, List

import numpy as np

from .general_transforms import *
from .rec_transforms import *

__all__ = ["create_transforms", "run_transforms"]
_logger = logging.getLogger(__name__)


def create_transforms(transform_pipeline: List, global_config: Dict = None):
    """
    Create a sequence of callable transforms.

    Args:
        transform_pipeline (List): list of callable instances or dicts where each key is a transformation class name,
            and its value are the args.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                 [DecodeImage(img_mode='BGR')]

    Returns:
        list of data transformation functions
    """
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
            # TODO: assert undefined transform class

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
