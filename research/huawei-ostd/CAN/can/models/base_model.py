from addict import Dict

from mindspore import nn

from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck

__all__ = ["BaseModel"]


class BaseModel(nn.Cell):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): model config

        Inputs:
            x (Tensor): The input tensor feeding into the backbone, neck and head sequentially.
            y (Tensor): The extra input tensor. If it is provided, it will feed into the head. Default: None
        """
        super(BaseModel, self).__init__()

        config = Dict(config)

        backbone_name = config.backbone.pop("name")
        self.backbone = build_backbone(backbone_name, **config.backbone)

        assert hasattr(self.backbone, "out_channels"), (
            f"Backbones are required to provide out_channels attribute, "
            f"but not found in {backbone_name}"
        )
        if "neck" not in config or config.neck is None:
            neck_name = "Select"
        else:
            neck_name = config.neck.pop("name")

        self.neck = build_neck(
            neck_name, in_channels=self.backbone.out_channels, **config.neck
        )

        assert hasattr(self.neck, "out_channels"), (
            f"Necks are required to provide out_channels attribute, "
            f"but not found in {neck_name}"
        )

        self.head_name = config.head.pop("name")
        self.head = build_head(self.head_name, in_channels=self.neck.out_channels, **config.head)

        self.model_name = f"{backbone_name}_{neck_name}_{self.head_name}"

    def ser(self, *inputs):
        input_ids, bbox, attention_mask, token_type_ids = inputs[:4]
        image = inputs[4] if self.backbone.use_visual_backbone else None

        x = self.backbone(input_ids, bbox, attention_mask, token_type_ids, image)
        x = self.head(x, input_ids)
        return x

    def re(self, *inputs):
        (input_ids, bbox, attention_mask, token_type_ids, question, question_label, answer, answer_label) = inputs[:8]
        image = inputs[8] if self.backbone.use_visual_backbone else None

        x = self.backbone(input_ids, bbox, attention_mask, token_type_ids, image)
        x = self.head(x, input_ids, question, question_label, answer, answer_label)
        return x

    def construct(self, *args):

        x = args[0]

        bout = self.backbone(x)

        nout = self.neck(bout)

        if len(args) > 1:
            hout = self.head(nout, args[1:])
        else:
            hout = self.head(nout)

        return hout
