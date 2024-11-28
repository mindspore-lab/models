import argparse
import os

import mindspore as ms
from mindspore import mint

from segment_anything import sam_model_registry


def main(args):
    # Step0: prepare
    ms.set_context(device_target='CPU')  # put mindspore on CPU to avoid competing for ascend memory with mindspore_lite
    os.makedirs(args.model_path_wo_ext, exist_ok=True)
    model_path = os.path.join(args.model_path, 'sam_' + args.model_type)
    mindir_path = os.path.join(model_path + '.mindir')
    # model
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    # Step 1: export mindir
    image = mint.ones(size=(1, 3, 1024, 1024), dtype=ms.float32)  # b, 3, 1024, 1024
    boxes = mint.ones(size=(1, 1, 4), dtype=ms.float32)  # b, n, 4
    inputs = (image, boxes)
    model.set_inputs(*inputs)
    print(f"start export mindir")
    ms.export(model, *inputs, file_name=model_path, file_format="MINDIR")
    print(f"finish export mindir")
    print(f'mind ir path: {mindir_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Export online ckpt to offline mindir"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default='./models/',
        help=(
            "Full path to the directory where the output model is saved, without file extension."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default='models/sam_vit_b-35e4849c.ckpt',
        help="online checkpoint file that stores weight",
    )

    parser.add_argument("--device", type=str, default="Ascend", help="The device to run generation on.")

    args = parser.parse_args()
    print(args)
    main(args)
