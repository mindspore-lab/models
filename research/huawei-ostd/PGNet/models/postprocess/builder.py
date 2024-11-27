from . import e2e_pg_postprocess
from .e2e_pg_postprocess import *

__all__ = ["build_postprocess"]

supported_postprocess = (e2e_pg_postprocess.__all__)


def build_postprocess(config: dict):
    proc = config.pop("name")
    if proc in supported_postprocess:
        postprocessor = eval(proc)(**config)
    elif proc is None:
        return None
    else:
        raise ValueError(f"Invalid postprocess name {proc}, support postprocess are {supported_postprocess}")

    return postprocessor
