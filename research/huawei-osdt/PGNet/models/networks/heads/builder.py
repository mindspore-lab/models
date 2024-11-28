__all__ = ['build_head']
supported_heads = ['PGNetHead']

from .e2e_head import PGNetHead


def build_head(head_name, **kwargs):
    """
    Build Head network.

    Args:
        head_name (str): the head layer(s) name, which shoule be one of the supported_heads.
        kwargs (dict): input args for the head network

    Return:
        nn.Cell for head module

    Construct:
        Input: Tensor
        Output: Dict[Tensor]
    """
    assert head_name in supported_heads, f'Invalid head {head_name}. Supported heads are {supported_heads}'
    head = eval(head_name)(**kwargs)
    return head
