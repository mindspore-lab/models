from typing import Dict

from mindspore import nn

def create_loss_scaler(args):
    assert 'type' in args
    scale_type = args.type
    if scale_type == 'fixed':
        value = args.get('loss_scale_value', 1.0)
        loss_scaler = nn.FixedLossScaleUpdateCell(value)
    elif scale_type == 'dynamic':
        loss_scaler = nn.DynamicLossScaleUpdateCell(
            loss_scale_value=args.get("loss_scale_value", 2 ** 12),
            scale_factor=args.get("scale_factor", 2.0),
            scale_window=args.get("scale_window", 1000),
        )
    else:
        raise NotImplementedError(f'Expected loss scaler in [fixed, dynamic], but got {scale_type}')

    return loss_scaler
