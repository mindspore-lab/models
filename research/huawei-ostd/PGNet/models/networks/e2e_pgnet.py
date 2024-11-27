from ._registry import register_model
from .base_model import BaseModel

__all__ = ['E2eNet', 'pgnet_resnet50']


class E2eNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def pgnet_resnet50(pretrained=False, pretrained_backbone=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'pgnet_backbone',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'E2eFpn'
        },
        "head": {
            "name": 'PGNetHead'
        }
    }
    model = E2eNet(model_config)

    return model