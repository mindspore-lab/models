from typing import Optional

from mindspore import nn


def create_optimizer(
    model_or_params,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    **kwargs,
):
    if isinstance(model_or_params, nn.Cell):
        params = model_or_params.trainable_params()
    else:
        params = model_or_params

    opt = opt.lower()
    opt_args = dict(**kwargs)

    if opt == "adam":
        optimizer = nn.Adam(
            params=params,
            learning_rate=lr,
            **opt_args,
        )
    else:
        raise NotImplementedError

    return optimizer
