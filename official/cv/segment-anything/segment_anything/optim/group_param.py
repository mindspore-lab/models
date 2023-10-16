def create_group_param(params,  **kwargs):
    """
    Create group parameters for optimizer.

    Args:
        params: Network parameters
        gp_weight_decay: Weight decay. Default: 0.0
        **kwargs: Others
    """
    # currently no parameter grouping strategy
    return params

