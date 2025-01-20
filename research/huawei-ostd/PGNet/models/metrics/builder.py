from . import e2e_metrics

from .e2e_metrics import *

__all__ = ["build_metric"]

supported_metrics = (e2e_metrics.__all__)

def build_metric(config, device_num=1, **kwargs):
    """
    Create the metric function.

    Args:
        config (dict): configuration for metric including metric `name` and also the kwargs specifically for
            each metric.
            - name (str): metric function name, exactly the same as one of the supported metric class names
        device_num (int): number of devices. If device_num > 1, metric will be computed in distributed mode,
            i.e., aggregate intermediate variables (e.g., num_correct, TP) from all devices
            by `ops.AllReduce` op so as to correctly
            compute the metric on dispatched data.

    Return:
        nn.Metric
    """
    mn = config.pop("name")
    if mn in supported_metrics:
        device_num = 1 if device_num is None else device_num
        config.update({"device_num": device_num})
        if "save_dir" in kwargs:
            config.update({"save_dir": kwargs["save_dir"]})
        metric = eval(mn)(**config)
    else:
        raise ValueError(f"Invalid metric name {mn}, support metrics are {supported_metrics}")

    return metric
