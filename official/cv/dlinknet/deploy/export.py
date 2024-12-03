import os.path

import numpy as np
from mindspore import load_checkpoint, load_param_into_net, export, Tensor, context
from models.dlinknet import DLinkNet34, DLinkNet50
from config import parse_args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    trained_ckpt_path = args.trained_ckpt
    local_train_url = './'

    BATCH_SIZE = args.batch_size

    # set context
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    # --------------------------------------
    if args.model_name == 'dlinknet34':
        net = DLinkNet34()
    else:
        net = DLinkNet50()
    param_dict = load_checkpoint(trained_ckpt_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([BATCH_SIZE, args.num_channels, args.width, args.height], np.float32))
    export(net, input_arr, file_name=os.path.join(local_train_url, args.file_name), file_format=args.file_format)
