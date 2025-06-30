"""
Training script for CMT-Small model on distributed platform
Dataset: ImageNet1k
Model: CMT-Small
Training method: Distributed training (4 NPUs)
"""

import os
import argparse
import mindspore
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore.common import set_seed

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler

from c2net.context import prepare, upload_output

parser = argparse.ArgumentParser(description='MindCV CMT-Small Training on ImageNet')

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'GPU', 'CPU'],
    help='device where the code will be implemented (default: Ascend)'
)

parser.add_argument('--epoch_size',
                    type=int,
                    default=300,
                    help='Training epochs.')

parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='Batch size per device.')

parser.add_argument('--distribute',
                    action='store_true',
                    default=True,
                    help='Run distribute training.')

def get_train_config():
    """Get training configuration"""
    config = {
        'model': 'cmt_small',
        'num_classes': 1000,
        'pretrained': False,
        
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.05,
        
        'scheduler': 'cosine_decay',
        'min_lr': 1e-6,
        'warmup_epochs': 20,
        'decay_epochs': 280,
        
        'amp_level': 'O2',
        'loss_scale': 1024,
        'label_smoothing': 0.1,
        
        'auto_augment': 'randaug-m9-mstd0.5',
        'mixup': 0.2,
        'cutmix': 1.0,
        
        'drop_path_rate': 0.1,
    }
    return config

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    
    c2net_context = prepare()
    
    imagenet_path = c2net_context.dataset_path + "/imagenet"
    
    output_path = c2net_context.output_path
    
    set_seed(42)
    
    if args.distribute:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        context.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True
        )
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        device_num = 1
        rank_id = 0
    
    config = get_train_config()
    
    print(f"Creating dataset from {imagenet_path}")
    train_path = os.path.join(imagenet_path, 'train')
    
    transform_list = create_transforms(
        dataset_name='imagenet',
        is_training=True,
        image_resize=224,
        auto_augment=config['auto_augment'],
        mixup=config['mixup'],
        cutmix=config['cutmix'],
    )
    
    dataset = create_dataset(
        name='imagenet',
        root=train_path,
        split='train',
        shuffle=True,
    )
    
    loader_train = create_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_remainder=True,
        is_training=True,
        transform=transform_list,
        num_parallel_workers=8,
        distribute=args.distribute
    )
    
    steps_per_epoch = loader_train.get_dataset_size()
    
    print(f"Creating CMT-Small model")
    network = create_model(
        model_name=config['model'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        drop_path_rate=config['drop_path_rate'],
    )
    
    loss = create_loss(
        name='CE',
        label_smoothing=config['label_smoothing'],
        reduction='mean'
    )
    
    lr_scheduler = create_scheduler(
        steps_per_epoch=steps_per_epoch,
        scheduler=config['scheduler'],
        lr=config['lr'] * device_num,
        min_lr=config['min_lr'],
        warmup_epochs=config['warmup_epochs'],
        warmup_factor=0.1,
        decay_epochs=config['decay_epochs'],
        epochs=args.epoch_size,
    )
    
    opt = create_optimizer(
        params=network.trainable_params(),
        opt=config['optimizer'],
        lr=lr_scheduler,
        weight_decay=config['weight_decay'],
    )
    
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_cb = LossMonitor()
    
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch * 10,
        keep_checkpoint_max=10
    )
    
    save_dir = os.path.join(output_path, f"rank_{rank_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_cb = ModelCheckpoint(
        prefix=f"cmt_small",
        directory=save_dir,
        config=ckpt_config
    )
    
    model = Model(
        network=network,
        loss_fn=loss,
        optimizer=opt,
        amp_level=config['amp_level'],
        loss_scale_manager=mindspore.FixedLossScaleManager(config['loss_scale'], False),
    )
    
    print("============== Starting Training ==============")
    print(f"Model: CMT-Small")
    print(f"Epochs: {args.epoch_size}")
    print(f"Batch size: {args.batch_size} per device")
    print(f"Total batch size: {args.batch_size * device_num}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training on {device_num} devices")
    
    callback_list = [time_cb, loss_cb]
    if rank_id == 0:
        callback_list.append(ckpt_cb)
    
    model.train(
        epoch=args.epoch_size,
        train_dataset=loader_train,
        callbacks=callback_list,
        dataset_sink_mode=True
    )
    
    upload_output()