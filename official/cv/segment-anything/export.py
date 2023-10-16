import argparse
import os

import mindspore_lite as mslite
import mindspore as ms
from mindspore import ops

from segment_anything import sam_model_registry


def main(args):
    # Step0: prepare
    ms.set_context(device_target='CPU')  # put mindspore on CPU to avoid competing for ascend memory with mindspore_lite
    os.makedirs(args.model_path_wo_ext, exist_ok=True)
    model_path_wo_ext = os.path.join(args.model_path_wo_ext, 'sam_' + args.model_type)
    mindir_path = os.path.join(model_path_wo_ext + '.mindir')
    lite_path_wo_ext = os.path.join(model_path_wo_ext + f"_lite")
    lite_path = os.path.join(model_path_wo_ext + f"_lite.mindir")
    # model
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    # Step 1: export mindir
    if args.export_mindir:
        # input
        image = ops.ones(shape=(1, 3, 1024, 1024), dtype=ms.float32)  # b, 3, 1024, 1024
        boxes = ops.ones(shape=(1, 1, 4), dtype=ms.float32)  # b, n, 4
        inputs = (image, boxes)
        model.set_inputs(*inputs)
        print(f"start export mindir")
        ms.export(model, *inputs, file_name=model_path_wo_ext, file_format="MINDIR")
        print(f"finish export mindir")

    print(f'mind ir path: {mindir_path}')
    print(f'lite path wo_ext: {lite_path_wo_ext}')
    print(f'lite path: {lite_path}')

    # Step 2: convert lite
    if args.convert_lite:
        optimize_dict = {"ascend": "ascend_oriented", "gpu": "gpu_oriented", "cpu": "general"}
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = optimize_dict[args.device.lower()]

        print(f"start convert lite")
        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=mindir_path,
            output_file=lite_path_wo_ext,
            config_file="./configs/export_lite.cfg",
        )
        print(f"finish convert lite")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Export online ckpt to offline mindir"
        )
    )

    parser.add_argument(
        "--model_path_wo_ext",
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

    parser.add_argument(
        "--export-mindir",
        default=True,
        help=(
            "Button to enable export mindir."
        ),
    )

    parser.add_argument(
        "--convert-lite",
        default=True,
        help=(
            "Button to enable convert lite."
        ),
    )
    args = parser.parse_args()
    print(args)
    main(args)
