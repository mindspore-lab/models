# Copyright 2024 Huawei Technologies Co., Ltd
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
"""run dpo"""
import argparse
import os
# pylint: disable=W0611
from mindformers import Trainer, MindFormerConfig
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.core.context import build_context
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.tools.logger import logger
from mindformers.tools.utils import str2bool

from utils import DPODataset
from qwen_mwpo import Qwen2_5_DPO
from qwen_tokenizer import Qwen2_5Tokenizer
@cloud_monitor()
def main(task='text_generation',
         config=None,
         run_mode=None,
         seq_length=None,
         mode=None,
         use_parallel=None,
         device_id=None,
         ckpt=None,
         strategy=None,
         auto_trans_ckpt=None,
         resume=False,
         train_dataset='',
         eval_dataset='',
         predict_data='',
         max_length=512,
         remote_save_url=None,
         vocab_file=None,
         merges_file=None,
         batch_size=None):
    """main function."""

    assert os.path.exists(config) and config.endswith(('.yaml', '.yml'))

    # init config
    config = MindFormerConfig(os.path.realpath(config))
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if mode is not None:
        config.context.mode = mode
        if mode:
            config.recompute_config.recompute = False
    if run_mode is not None:
        config.run_mode = run_mode
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id
    if ckpt is None:
        ckpt = config.load_checkpoint
    if strategy is not None and os.path.exists(strategy):
        config.src_strategy_path_or_dir = strategy
    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
    if remote_save_url is not None:
        config.remote_save_url = remote_save_url
    if vocab_file is not None:
        config.processor.tokenizer.vocab_file = vocab_file
    if merges_file is not None:
        config.processor.tokenizer.merges_file = merges_file
    if batch_size is not None:
        config.runner_config.batch_size = batch_size

    # init context
    build_context(config)

    if run_mode in ['train', 'finetune']:
        config.model.model_config.use_past = False

    # start task
    if run_mode == 'train':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.train(train_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)
    elif run_mode == 'finetune':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)
    elif run_mode == 'eval':
        trainer = Trainer(args=config,
                          task=task,
                          eval_dataset=eval_dataset)
        trainer.evaluate(eval_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt)
    elif run_mode == 'predict':
        trainer = Trainer(args=config,
                          task=task)
        batch_input = [[predict_data for _ in range(config.model.model_config.batch_size)]]
        for input_prompt in batch_input:
            result = trainer.predict(input_data=input_prompt,
                                     predict_checkpoint=ckpt,
                                     auto_trans_ckpt=config.auto_trans_ckpt,
                                     max_length=int(max_length))
            logger.info(result)
    else:
        raise ValueError(f'run_mode should be one of [train, finetune, eval, predict], but get {config.run_mode}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default=None, type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model.')
    parser.add_argument('--seq_length', default=None, type=int,
                        help='seq_length')
    parser.add_argument('--use_parallel', default=None, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--device_id', default=None, type=int,
                        help='device id set when run on single card. Default: 0')
    parser.add_argument('--mode', default=0, type=int,
                        help='0--Graph Mode; 1--Pynative Mode')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--src_strategy', default=None, type=str,
                        help='strategy of load_checkpoint')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--resume', default=None, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--eval_dataset', default='', type=str,
                        help='set eval dataset.')
    parser.add_argument('--predict_data', default='', type=str, nargs='+',
                        help='input predict data.')
    parser.add_argument('--max_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--remote_save_url', default='', type=str,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--vocab_file', default=None, type=str,
                        help='tokenizer model or vocab_file')
    parser.add_argument('--merges_file', default=None, type=str,
                        help='merges_file')
    parser.add_argument('--batch_size', default=None, type=str,
                        help='batch_size')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         seq_length=args.seq_length,
         mode=args.mode,
         use_parallel=args.use_parallel,
         device_id=args.device_id,
         ckpt=args.load_checkpoint,
         strategy=args.src_strategy,
         auto_trans_ckpt=args.auto_trans_ckpt,
         resume=args.resume,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         max_length=args.max_length,
         remote_save_url=args.remote_save_url,
         vocab_file=args.vocab_file,
         merges_file=args.merges_file,
         batch_size=args.batch_size)
