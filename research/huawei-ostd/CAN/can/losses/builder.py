from .rec_loss import CANLoss

__all__ = ["build_loss"]

supported_losses = [
    "CANLoss",
]


def build_loss(name, **kwargs):
    """
    Create the loss function.

    Args:
        name (str): loss function name, exactly the same as one of the supported loss class names

    Return:
        nn.LossBase
    """
    assert name in supported_losses, f"Invalid loss name {name}, support losses are {supported_losses}"

    loss_fn = eval(name)(**kwargs)

    # print('=> Loss func input args: \n\t', inspect.signature(loss_fn.construct))

    return loss_fn
