import os
import argparse
import ast
from logzero import logger
from mindspore import nn
from mindspore import Model
from mindspore import ops
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, LossMonitor, TimeMonitor, CheckpointConfig
from mindvision.engine.callback import ValAccMonitor
from mindspore import context
import mindspore as ms
from dataset import create_cifar_dataset
from model.mlp import ConvMLP
from mindspore.nn.metrics import Accuracy
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

parser = argparse.ArgumentParser(description='Train CMLPNet')

parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='train modelarts')
parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help="use 8 npus")
parser.add_argument('--device_id', type=int, default=4)
parser.add_argument("--lr", default=0.001)#
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch_size', type=int, default=128)
parser.add_argument('--dataset_choose', type=str, default='cifar10', help="cifar10 or cifar100")
parser.add_argument('--device_target', type=str, default='Ascend')
parser.add_argument('--checkpoint_path',
                    type=str,
                    default="./checkpoint/cifa10_CMLP.ckpt",
                    )
parser.add_argument('--save_checkpoint_path',
                    type=str,
                    default="./ckpt",
                    help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()



if __name__ == '__main__':

    dataset_choose=args.dataset_choose
    image_height, image_width = 224, 224
    if dataset_choose=='cifar10':
        class_num=10
    elif dataset_choose=='cifar100':
        class_num = 100
    args.save_checkpoint_path = './'
    mode = None
    if args.is_distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE', '1'))
        context.set_context(device_id=device_id)
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        init()
        rank_id = get_rank()
        ckpt_save_dir = os.path.join(args.save_checkpoint_path, "ckpt_" + str(rank_id) + "/")
    else:
        device_id = int(os.getenv('DEVICE_ID','0'))
        context.set_context(device_id=device_id)
        rank_id = 0
        device_num = 1
        ckpt_save_dir = os.path.join(args.save_checkpoint_path,'./')
    # profiles = Profiler()

    if mode is None or mode == 'train':


        train_dataset = create_cifar_dataset( True,dataset_choose, batch_size=args.batch_size,
                                             image_size=(int(image_height), int(image_width)),
                                             mixup=False)
        train_mixup_dataset = create_cifar_dataset(True, dataset_choose, batch_size=args.batch_size,
                                             image_size=(int(image_height), int(image_width)),
                                             mixup=True)

        valid_dataset = create_cifar_dataset( False,dataset_choose, batch_size=args.batch_size,
                                            image_size=(int(image_height), int(image_width)))

        train_ds = train_dataset
        train_dsmix = train_mixup_dataset
        #valid

        valid_ds = valid_dataset


        network = ConvMLP(blocks=[2, 4, 2], dims=[128, 256, 512], mlp_ratios=[2, 2, 2],
                     classifier_head=True, channels=64, n_conv_blocks=2,num_classes=class_num)
        ms.load_checkpoint(args.checkpoint_path, network)
        loss_fn = nn.CrossEntropyLoss() # 定义损失函数



        # valid_net = EvalNet(network, k=5)
        lr = args.lr
        optimizer = nn.SGD(params=network.trainable_params(), learning_rate=lr,momentum=0.9,
                           weight_decay=0.005,nesterov=True)
        # scheduler=nn.cosine_decay_lr(min_lr=1e-5,max_lr=lr,step_per_epoch=2,decay_epoch=2)
        save_steps = train_ds.get_dataset_size()

        time_cb = TimeMonitor()
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if rank_id == 0:
            config_ckp = CheckpointConfig(save_checkpoint_steps=save_steps, keep_checkpoint_max=20)
            ckpt_cb = ModelCheckpoint(prefix='corenet', directory=ckpt_save_dir,
                                      config=config_ckp)
            cb.append(ckpt_cb)
            # val=ValAccMonitor(network, valid_ds, 100)
            # cb.append(val)

        model = Model(network=network,loss_fn=loss_fn,optimizer=optimizer, metrics={"acc"},amp_level="O3")

        num_epochs = 100
        logger.info(F"epoch size: {num_epochs}")
        for i in range(num_epochs):
            model.train(1, train_ds, callbacks=cb, dataset_sink_mode=True)
            model.train(1, train_dsmix, callbacks=cb, dataset_sink_mode=True)
            acc=model.eval(valid_dataset)
            print('-------------------------------------')
            print(acc)
            print('-------------------------------------')
        logger.info('Finish Training')
