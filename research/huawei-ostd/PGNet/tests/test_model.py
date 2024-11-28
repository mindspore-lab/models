import numpy as np
import mindspore as ms

from models.networks.backbones import build_backbone
from models.networks.necks import build_neck
from models.networks.heads import build_head

def test_model():
    model_config = {
        "model_type": "e2e",
        "backbone": {
            "name": "pgnet_backbone",
            "pretrained": False,
        },
        "neck": {
            "name": 'E2eFpn',
            "in_channels": [3, 64, 256, 512, 1024, 2048, 2048],
        },
        "head": {
            "name": 'PGNetHead',
            "in_channels": 128,
        },
    }
    batch_size = 1
    input_tensor_channel = 3
    h = 448
    w = 448

    numpy_array = np.ones((batch_size, input_tensor_channel, h, w), dtype=np.float32)

    input_tensor = ms.Tensor(numpy_array, ms.float32)

    backbone_name = model_config["backbone"].pop("name")
    backbone = build_backbone(backbone_name, **model_config["backbone"])
    bout = backbone(input_tensor)
    assert bout[1].shape == (batch_size, 64, 224, 224)
    assert bout[2].shape == (batch_size, 256, 112, 112)
    assert bout[3].shape == (batch_size, 512, 56, 56)
    assert bout[4].shape == (batch_size, 1024, 28, 28)
    assert bout[5].shape == (batch_size, 2048, 14, 14)
    assert bout[6].shape == (batch_size, 2048, 7, 7)

    neck_name = model_config["neck"].pop("name")
    neck = build_neck(neck_name, **model_config["neck"])
    nout = neck(bout)
    assert nout.shape == (batch_size, 128, 112, 112)

    head_name = model_config["head"].pop("name")
    head = build_head(head_name, **model_config["head"])
    hout = head(nout)
    assert hout['f_score'].shape ==(batch_size, 1, 112, 112)
    assert hout['f_border'].shape ==(batch_size, 4, 112, 112)
    assert hout['f_char'].shape ==(batch_size, 37, 112, 112)
    assert hout['f_direction'].shape ==(batch_size, 2, 112, 112)
