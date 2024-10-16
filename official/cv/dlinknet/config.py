import argparse
import logging
import os

import yaml

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# fmt: off
def create_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='./configs/dlinknet34_config.yaml',
                               help='YAML config file specifying default arguments (default="./configs/dlinknet34_config.yaml")')

    # The main parser. It inherits the --config argument for better help information.
    parser = argparse.ArgumentParser(description='dlinknet Training', parents=[parser_config])

    # System parameters
    group = parser.add_argument_group('System parameters')
    group.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    group.add_argument('--run_distribute', type=str2bool, nargs='?', const=True, default=False,
                       help='Run distribute (default=False)')
    group.add_argument('--jit_level', type=str, default='O2',
                       help='Used to control the compilation optimization level. Supports ["O0", "O1", "O2"]')
    group.add_argument('--device_target', type=str, default='Ascend',
                       help='Device target for computing, currently only supports Ascend')

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--data_dir', type=str,
                       help='Path to dataset')
    group.add_argument('--batch_size', type=int, default=4,
                       help='Number of batch size (default=4)')
    group.add_argument('--num_parallel_workers', type=int, default=4,
                       help='Number of parallel workers (default=4)')
    
    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model_name', type=str, default='dlinknet34',
                       help='Name of model')
    group.add_argument('--pretrained_ckpt', type=str,
                       help='Initialize model from this resnet checkpoint. ')
    group.add_argument('--output_path', type=str, default="./output",
                       help='Path of checkpoint (default="./output")')
    group.add_argument('--epoch_size', type=int, default=300,
                       help='Train epoch size (default=300)')

    # Optimize parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', type=str, default='adam',
                       choices=['adam'],
                       help='Type of optimizer (default="adam")')
    
    # Scheduler parameters
    group = parser.add_argument_group('Scheduler parameters')
    group.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate (default=0.0002)')

    # Loss parameters
    group = parser.add_argument_group('Loss parameters')
    group.add_argument('--loss', type=str, default='bce', choices=['bce'],
                       help='Type of loss, bce (BinaryCrossEntropy)  (default="bce")')

    # AMP parameters
    group = parser.add_argument_group('Auto mixing precision parameters')
    group.add_argument('--amp_level', type=str, default='O0',
                       help='Amp level - Auto Mixed Precision level for saving memory and acceleration. '
                            'Choice: O0 - all FP32, O1 - only cast ops in white-list to FP16, '
                            'O2 - cast all ops except for blacklist to FP16, '
                            'O3 - cast all ops to FP16. (default="O0").')
    group.add_argument('--dataset_sink_mode', type=str2bool, nargs='?', const=True, default=True,
                       help='The dataset sink mode (default=True)')

    # EVAL parameters
    group = parser.add_argument_group('Eval parameters')
    group.add_argument('--label_path', type=str,
                       help='The label path of the valid dataset')
    group.add_argument('--trained_ckpt', type=str,
                       help='The trained checkpoint file path')
    group.add_argument('--predict_path', type=str,
                       help='The storage path for predicted results')
    
    # EXPORT parameters
    group = parser.add_argument_group('Export parameters')
    group.add_argument('--file_name', type=str,
                       help='The file name of mindspore mindir')
    group.add_argument('--file_format', type=str, default="MINDIR",
                       help='The file format export')

    return parser_config, parser

def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args(args=None):
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args(args)
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def save_args(args: argparse.Namespace, filepath: str, rank: int = 0) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.
    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(args, argparse.Namespace)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(args.__dict__, f)
    logger.info(f"Args is saved to {filepath}.")
