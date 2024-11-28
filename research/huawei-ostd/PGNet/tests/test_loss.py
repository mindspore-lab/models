import mindspore as ms
from mindspore import ops
from models.losses import build_loss

def test_loss():
    loss_func_name = "PGLoss"
    loss_func_config = {
        "tcl_bs": 64, 
        "max_text_length": 50, 
        "max_text_nums": 30, 
        "pad_num": 36
    }
    loss_fn = build_loss(loss_func_name, **loss_func_config)

    f_score = ops.ones((1, 1, 128, 128), ms.float32)
    f_border = ops.ones((1, 4, 128, 128), ms.float32)
    f_char = ops.ones((1, 37, 128, 128), ms.float32)
    f_direction = ops.ones((1, 2, 128, 128), ms.float32)
    predicts = {
        "f_score": f_score, 
        "f_border": f_border, 
        "f_char": f_char, 
        "f_direction": f_direction
    }

    tcl_maps = ops.zeros((1, 1, 128, 128), ms.float32)
    tcl_label_maps = ops.zeros((1, 1, 128, 128), ms.float32)
    border_maps = ops.zeros((1, 5, 128, 128), ms.float32)
    direction_maps = ops.ones((1, 3, 128, 128), ms.float64)
    training_masks = ops.ones((1, 1, 128, 128), ms.float32)
    label_list = ops.ones((1, 30, 50, 1), ms.float64)
    pos_list = ops.functional.full((1, 30, 64, 3), 0.5, dtype=ms.float64)
    pos_mask = ops.ones((1, 30, 64, 1), ms.float64)

    loss = loss_fn(
        predicts, 
        tcl_maps, 
        tcl_label_maps, 
        border_maps, 
        direction_maps, 
        training_masks, 
        label_list, 
        pos_list, 
        pos_mask
    )

    assert loss, "loss is empty"
