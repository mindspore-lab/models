from . import (  
    rec_postprocess,
)

from .rec_postprocess import *

__all__ = ["build_postprocess"]

supported_postprocess = (
    rec_postprocess.__all__
)


def build_postprocess(config: dict):
    """
    Create postprocess function.

    Args:
        config (dict): configuration for postprocess including postprocess `name` and also the kwargs specifically
        for each postprocessor.
            - name (str): metric function name, exactly the same as one of the supported postprocess class names

    Return:
        Object

    Example:
        >>> # Create postprocess function
        >>> from mindocr.postprocess import build_postprocess
        >>> config = dict(name="RecCTCLabelDecode", use_space_char=False)
        >>> postprocess = build_postprocess(config)
        >>> postprocess
    """
    proc = config.pop("name")
    if proc in supported_postprocess:
        postprocessor = eval(proc)(**config)
    elif proc is None:
        return None
    else:
        raise ValueError(f"Invalid postprocess name {proc}, support postprocess are {supported_postprocess}")

    return postprocessor
