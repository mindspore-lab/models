__all__ = ['build_neck']
supported_necks = ['E2eFpn']
from .e2e_neck import E2eFpn


def build_neck(neck_name, **kwargs):
    """
    Build Neck network.

    Args:
        neck_name (str): the neck name, which shoule be one of the supported_necks.
        kwargs (dict): input args for the neck network

    Return:
        nn.Cell for neck module
    """
    assert neck_name in supported_necks, f'Invalid neck: {neck_name}, Support necks are {supported_necks}'
    neck = eval(neck_name)(**kwargs)
    return neck
