from .pg_loss import PGLoss

__all__ = ["build_loss"]

supported_losses = ["PGLoss"]


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
    return loss_fn
