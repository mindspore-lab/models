from mixtral.mixtral import run_umoe
import argparse
from utils import str2bool
import os 
import mindspore as ms
import time
def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/cache/transformers",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/cache/offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=10)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-batch-size", type=int, default= 8)
    parser.add_argument("--num-gpu-batches", type=int, default= 8)
    parser.add_argument("--manual_offload", nargs="+", type=int,
        default=[-1, -1, -1],
        help="Three numbers: [expert_w, dense_w, kvcache]"
            "0: GPU, 1:CPU, 2:DISK")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=False, default=False)
    parser.add_argument("--attn-sinkToken", action="store_true")
    parser.add_argument("--HQQ-quantize", action="store_true")
    parser.add_argument("--HQQ-config", nargs="+", type=int,
        default=[0, 0, 0, 0],
        help="Four numbers. They are: [attn_num_bits, attn_group_size ,expert_num_bits, expert_group_size]")
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--store-output", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--overlap", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--device", type=int, default=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    if "/" in args.model:
        model_name = args.model.split("/")[1].lower()
    args.path = os.path.dirname(os.getcwd()) + args.path + f"/{model_name}"
    args.offload_dir = os.path.dirname(os.getcwd()) + args.offload_dir

    # import mindspore.context as context
    # context.set_context(mode=context.PYNATIVE_MODE)
    # ms.set_context(memory_optimize_level='O1')
    # ms.set_context(max_device_memory="30GB")

    if args.model == "mistralai/Mixtral-8x7B-v0.1":
        run_umoe(args)
    elif args.model == "mistralai/Mixtral-8x22B-v0.1":
        run_umoe(args)
    else:
        print("Wrong model name")
 
