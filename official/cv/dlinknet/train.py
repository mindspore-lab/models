import os
import time
import json
import mindspore
import mindspore.dataset as ds
import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import TimeMonitor

from config import parse_args
from models.model_factory import create_model
from optim.optim_factory import create_optimizer
from loss.loss_factory import create_loss
from model_utils.trainer_factory import create_trainer
from data.data import ImageFolderGenerator
from model_utils.callback import MyCallback


def create_dataset(data_dir, shuffle, device_num, shard_id, num_parallel_workers, batch_size):
    """
    when doing distributed training, dataset.GeneratorDataset need to set num_shards and shard_id
    see:
    https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset
    https://blog.csdn.net/weixin_43431343/article/details/121895510
    """
    image_list = filter(lambda x: x.find('sat') != -1, os.listdir(data_dir))
    train_list = list(map(lambda x: x[:-8], image_list))
    dataset_generator = ImageFolderGenerator(train_list, data_dir)
    if device_num == 1:
        _dataset = ds.GeneratorDataset(dataset_generator,
                                       ["img", "mask"],
                                       shuffle=shuffle,
                                       num_parallel_workers=num_parallel_workers)
    else:
        _dataset = ds.GeneratorDataset(dataset_generator,
                                       ["img", "mask"],
                                       shuffle=shuffle,
                                       num_parallel_workers=num_parallel_workers,
                                       num_shards=device_num,
                                       shard_id=shard_id)

    _dataset = _dataset.batch(batch_size)  # set batch size
    return _dataset


if __name__ == "__main__":
    args = parse_args()
    args_str = json.dumps(vars(args), indent=4)
    print(args_str)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    time_start = time.time()
    if args.device_target not in ['Ascend']:
        raise Exception("Only support on Ascend currently.")

    # set context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        jit_config={"jit_level": args.jit_level})
    epoch_num = args.epoch_num
    if args.run_distribute:
        init()
        device_num = get_group_size()
        epoch_num = args.distribute_epoch_num
        print("group_size(device_num) is: ", device_num)
        rank_id = get_rank()
        print("rank_id is: ", rank_id)
        # set auto parallel context
        context.set_auto_parallel_context(device_num=device_num,
                                          gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL
                                          )
    else:
        device_num = 1
        rank_id = 0
    mindspore.common.set_seed(2022)
    local_data_url = args.data_dir
    pretrained_ckpt_path = args.pretrained_ckpt

    # prepare weight file and log file
    log_name = args.model_name
    rank_label = '[' + str(rank_id) + ']'

    file_name = os.path.join(args.output_path, str(log_name) + "_rank" + str(rank_id) + ".ckpt")

    # create dataset
    dataset_train = create_dataset(
        data_dir=args.data_dir,
        shuffle=args.shuffle,
        device_num=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        batch_size=args.batch_size,
    )

    # create model
    network = create_model(args.model_name)
    network.set_train()

    # define optimizer
    optimizer = create_optimizer(model_or_params=network.trainable_params(), opt=args.opt, lr=learning_rate)

    # define loss
    loss = create_loss(name=args.loss)

    trainer = create_trainer(network=network, loss=loss, optimizer=optimizer, amp_level=args.amp_level,
                             loss_scale=args.init_loss_scale, scale_factor=args.scale_factor,
                             scale_window=args.scale_window)

    # callback
    myCallback = MyCallback(file_name, rank_label, device_num, show_step=False, lr=learning_rate, model_name=args.model_name)
    # train
    trainer.train(args.epoch_size, dataset_train, callbacks=[TimeMonitor(), myCallback],
                  dataset_sink_mode=args.dataset_sink_mode)

    time_end = time.time()
    print('train_time: %f hours' % ((time_end - time_start)/3600))
