import argparse
import ast

import mindspore as ms
from mindspore import amp

from segment_anything.build_sam import create_model
from segment_anything.dataset.dataset import create_dataloader
from segment_anything.modeling.loss import create_loss_fn
from segment_anything.optim.loss_scaler import create_loss_scaler
from segment_anything.optim.optimizer import create_optimizer
from segment_anything.utils import logger
from segment_anything.utils.callbacks import create_callback
from segment_anything.utils.config import parse_args
from segment_anything.utils.model_wrapper import NetWithLossWrapper, TrainOneStepCellWrapper
from segment_anything.utils.utils import set_distributed, set_directory_and_log, update_rank_to_dataloader_config


def main(args) -> None:
    # Step1: initialize environment
    ms.context.set_context(mode=args.mode, device_target=args.device, pynative_synchronize=False)
    ms.set_seed(42)

    rank_id, rank_size, main_device = set_distributed(args.distributed)
    update_rank_to_dataloader_config(rank_id, rank_size, args.train_loader, args.eval_loader, args.callback)

    set_directory_and_log(main_device, rank_id, rank_size, args.work_root, args.log_level, args.callback)
    logger.info(args.pretty())

    # Step2: create dataset
    train_dataloader = create_dataloader(args.train_loader)

    # create model, load pretrained ckpt, set amp level, also freeze layer if specified
    network = create_model(args.network.model)
    loss_fn = create_loss_fn(args.network.loss)
    network.set_train()
    network = amp.auto_mixed_precision(network, args.get('amp_level', 'O0'))

    # Step3: create optimizer, including learning rate scheduler and group parameter settings
    optimizer = create_optimizer(params=network.trainable_params(), args=args.optimizer)

    # Step4: wrap model and optimizer for training
    with_loss_model = NetWithLossWrapper(network, loss_fn=loss_fn,
                                         input_columns=[args.train_loader.model_column, args.train_loader.loss_column],
                                         all_columns=args.train_loader.dataset.output_column,
                                         )

    loss_scaler = create_loss_scaler(args.loss_manager.loss_scaler)
    model = TrainOneStepCellWrapper(with_loss_model, optimizer=optimizer, scale_sense=loss_scaler,
                                    drop_overflow_update=args.loss_manager.drop_overflow_update)

    # Step5: train model
    callbacks = create_callback(args.callback)
    model = ms.Model(model)
    model.train(epoch=args.train_loader.epoch_size, train_dataset=train_dataloader, callbacks=callbacks)


if __name__ == "__main__":
    parser_config = argparse.ArgumentParser(description="SAM Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="configs/coco_box_finetune.yaml",
        help="YAML config file specifying default arguments."
    )
    parser_config.add_argument('-o', '--override-cfg', nargs='+',
                               help="command line to override configuration in config file."
                                    "For dict, use key=value format, eg: device=False. "
                                    "For nested dict, use '.' to denote hierarchy, eg: optimizer.weight_decay=1e-3."
                                    "For list, use number to denote position, eg: callback.1.interval=100.")

    # model arts
    parser_config.add_argument("--enable_modelarts", type=ast.literal_eval, default=False)
    parser_config.add_argument("--train_url", type=str, default="", help="obs path to output folder")
    parser_config.add_argument("--data_url", type=str, default="", help="obs path to dataset folder")

    args = parse_args(parser_config)
    main(args)
