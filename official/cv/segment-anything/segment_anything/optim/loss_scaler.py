from mindspore import nn, amp


def create_loss_scaler(args):
    assert 'type' in args
    scale_type = args.type
    use_amp = args.get('use_amp_scale', False)
    if scale_type == 'fixed':
        value = args.get('loss_scale_value', 1.0)
        func = amp.StaticLossScaler if use_amp else nn.FixedLossScaleUpdateCell
        loss_scaler = func(value)
    elif scale_type == 'dynamic':
        func = amp.DynamicLossScaler if use_amp else nn.DynamicLossScaleUpdateCell
        loss_scaler = func(
            args.get("loss_scale_value", 2 ** 12),
            scale_factor=args.get("scale_factor", 2),
            scale_window=args.get("scale_window", 1000),
        )
    else:
        raise NotImplementedError(f'Expected loss scaler in [fixed, dynamic], but got {scale_type}')

    return loss_scaler
