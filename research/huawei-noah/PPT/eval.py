# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluation script."""

from functools import reduce

import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore import Parameter

from src.args import get_args
from src.model.factory import create_model
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import set_device, load_pretrained, \
    get_train_one_step
from src.tools.optimizer import get_optimizer

from src.data.imagenet import create_datasets
from src.model.layers.tome import benchmark

def main():
    args = get_args()
    set_seed(args.seed)
    if args.mode == 'GRAPH_MODE':
        mode = context.GRAPH_MODE
    else:
        mode = context.PYNATIVE_MODE
    
    # for debug
    mode = context.PYNATIVE_MODE
    
    context.set_context(
        mode=mode, device_target=args.device_target
    )
    
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_reduce_precision=True)
    set_device(args.device_target, args.device_id)

    net = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )
    
    # runs = 50
    # batch_size = 256  # Lower this if you don't have that much memory
    # input_size = [3, 224, 224]
    # ppt_throughput = benchmark(net, input_size=input_size, batch_size=batch_size, runs=runs)

    cast_amp(net, args.amp_level, args)
    net.set_train(False)

    if not args.finetune:
        raise RuntimeError('Path to checkpoint (pretrained option) not set.')
    if args.finetune:
        load_pretrained(
            net, args.finetune, args.num_classes, args.exclude_epoch_state
        )

    print(
        'Number of parameters:',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    criterion = get_criterion(
        smoothing=args.smoothing,
        num_classes=args.num_classes,
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_minmax=args.cutmix_minmax,
        bce_loss=args.bce_loss,
        distillation_type=args.distillation_type,
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        distillation_alpha=args.distillation_alpha,
        distillation_tau=args.distillation_tau
    )
    net_with_loss = NetWithLoss(net, criterion)
    _, val_dataset = create_datasets(args)

    batch_num = val_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    eval_network = nn.WithEvalCell(
        net, nn.CrossEntropyLoss(), args.amp_level in ["O2", "O3", "auto"]
    )
    eval_indexes = [0, 1, 2]
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    
    loss_monitor_cb = ms.LossMonitor(args.print_loss_every)
    print(f"=> begin eval")
    # breakpoint()
    results = model.eval(val_dataset, callbacks=[loss_monitor_cb])
    print(f"=> eval results: {results}")
    print(f"=> eval success")


if __name__ == '__main__':
    main()
