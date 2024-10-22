from mindspore import ops


def meshgrid(inputs, indexing="xy"):
    # An alternative implementation of ops.meshgrid, Only supports inputs with a length of 2.
    # Meshgrid op is not supported on a specific model of machine an alternative
    # solution is adopted, which will be updated later.
    x, y = inputs
    nx, ny = x.shape[0], y.shape[0]
    xv, yv = None, None
    if indexing == "xy":
        xv = ops.tile(x.view(1, -1), (ny, 1))
        yv = ops.tile(y.view(-1, 1), (1, nx))
    elif indexing == "ij":
        xv = ops.tile(x.view(-1, 1), (1, ny))
        yv = ops.tile(y.view(1, -1), (nx, 1))

    return xv, yv
