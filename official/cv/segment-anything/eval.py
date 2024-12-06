import argparse

import mindspore as ms
from mindspore import amp

from segment_anything.build_sam import create_model
from segment_anything.dataset.dataset import create_dataloader
from segment_anything.evaluate.evaluator import Evaluator
from segment_anything.evaluate.metrics import create_metric
from segment_anything.utils import logger
from segment_anything.utils.config import parse_args
from segment_anything.utils.utils import set_directory_and_log, set_distributed, update_rank_to_dataloader_config, set_env


def main(args) -> None:
    # initialize environment
    set_env(args)

    rank_id, rank_size, main_device = set_distributed(args.distributed)
    update_rank_to_dataloader_config(rank_id, rank_size, args.train_loader, args.eval_loader)

    set_directory_and_log(main_device, rank_id, rank_size, args.work_root, args.log_level)
    logger.info(args.pretty())

    # create dataset
    eval_dataloader = create_dataloader(args.eval_loader)

    # create model, also freeze layer if specified
    network = create_model(args.network.model)
    network.set_train(False)
    network = amp.auto_mixed_precision(network, args.get('amp_level', 'O0'))

    # create evaluator
    metric = create_metric(args.eval_metric)
    evaluator = Evaluator(network=network, data_loader=eval_dataloader, metric=metric,
                          input_column=[args.eval_loader.model_column, args.eval_loader.eval_column])
    # eval model
    evaluator.eval(args.eval_loader.max_eval_iter)


if __name__ == "__main__":
    parser_config = argparse.ArgumentParser(description="SAM Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="./configs/coco_box_finetune.yaml",
        help="YAML config file specifying default arguments."
    )
    parser_config.add_argument('-o', '--override-cfg', nargs='+', help="command config to override that in config file")
    args = parse_args(parser_config)
    main(args)
