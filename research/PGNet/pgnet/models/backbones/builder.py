from pgnet.models.load_model import load_model
from ._registry import backbone_class_entrypoint, backbone_entrypoint, is_backbone, is_backbone_class, list_backbones

__all__ = ['build_backbone']


def build_backbone(name, **kwargs):
    """
    Build the backbone network.

    Args:
        name (str): the backbone name, which can be a registered backbone class name
                        or a registered backbone (function) name.
        kwargs (dict): input args for the backbone
           1) if `name` is in the registered backbones (e.g. det_resnet50), kwargs include args for backbone creating
           likes `pretrained`
           2) if `name` is in the registered backbones class (e.g. DetResNet50), kwargs include args for the backbone
           configuration like `layers`.
           - pretrained: can be bool or str. If bool, load model weights from default url defined in the backbone py
           file. If str, pretrained can be url or local path to a checkpoint.


    Return:
        nn.Cell for backbone module
    """
    remove_prefix = kwargs.pop("remove_prefix", False)

    if is_backbone(name):
        create_fn = backbone_entrypoint(name)
        backbone = create_fn(**kwargs)
    elif is_backbone_class(name):
        backbone_class = backbone_class_entrypoint(name)
        backbone = backbone_class(**kwargs)
    else:
        raise ValueError(f'Invalid backbone name: {name}, supported backbones are: {list_backbones()}')

    if 'pretrained' in kwargs:
        pretrained = kwargs['pretrained']
        if not isinstance(pretrained, bool):
            if remove_prefix:
                # remove the prefix with `backbone.`
                def fn(x): return {k.replace('backbone.', ''): v for k, v in x.items()}
            else:
                fn = None
            load_model(backbone, pretrained, filter_fn=fn)

    return backbone
