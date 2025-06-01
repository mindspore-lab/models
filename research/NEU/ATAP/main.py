import json
import os
# import torch
import argparse
import numpy as np
# from ipdb import set_trace#打断点
from log import *


# from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
# from transformers import AutoTokenizer

from os.path import join, abspath, dirname

from data_utils.dataset import load_file,LAMADataset,LAMADataset_new
from data_utils.vocab import *
from p_tuning.modeling import *

# mindspore
import argparse
from os.path import join, abspath, dirname
import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore.dataset import GeneratorDataset
# from mindformers import AutoTokenizer
from mindspore import ops

import os
from os.path import join
from datetime import datetime
import mindspore
from mindspore import save_checkpoint, Tensor
import numpy as np

from mindspore import nn, ops
from mindspore.common.initializer import XavierUniform
from tqdm import tqdm
import mindspore as ms
from mindspore import save_checkpoint
# from mindnlp.transformers import BertTokenizer
from mindformers import BertTokenizer

SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large','bert_base_uncased']


def set_seed(args):
    np.random.seed(args.seed)
    ms.set_seed(args.seed)

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="ConceptNet")
    parser.add_argument("--model_name", type=str, default='bert-base-uncased', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(3,3,0)")
    parser.add_argument("--early_stop", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-5)  # 1e-5
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)  # 是否使用原始的模板
    parser.add_argument("--use_lm_finetune", type=bool, default=False)  # 是否使用语言模型进行微调

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), 'data/CN-100K'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), 'results_out'))
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()

    # post-parsing args
    if not args.no_cuda and ms.context.get_context("device_target") == "GPU":
        device_id = 1  # MindSpore uses device_id instead of cuda:1
        context.set_context(device_id=device_id)
        args.device = "GPU"
        args.n_gpu = 1  # MindSpore typically manages one device per process
    else:
        args.device = "CPU"
        args.n_gpu = 0

    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 1. 设备初始化 (MindSpore专用方式)
        context.set_context(
            mode=context.PYNATIVE_MODE, 
            device_target="CPU",
            device_id=1 if args.model_name != 't5-11b' else args.t5_shard * 4
        )

        if self.args.use_original_template and (not self.args.use_lm_finetune) and (not self.args.only_evaluate):
            raise RuntimeError("""If use args.use_original_template is True, either args.use_lm_finetune or args.only_evaluate should be True.""")
          
        # 2. 加载分词器 
        self.tokenizer_old = BertTokenizer.from_pretrained('PLM_MODELS/bert_base_uncased')
        _ , self.tokenizer= create_model(self.args)
        init_vocab(args) 
        self.data_path_pre, self.data_path_post = self.get_TREx_parameters()#获取训练数据集的路径以及后缀
        self.train_data = load_file(join(self.args.data_dir, self.data_path_pre + 'train' + self.data_path_post))#导入训练集、测试集以及验证集
        self.dev_data = load_file(join(self.args.data_dir, self.data_path_pre + 'dev' + self.data_path_post))
        self.test_data = load_file(join(self.args.data_dir, self.data_path_pre + 'test' + self.data_path_post))

        #开始
        self.test_set = LAMADataset('test', self.test_data, self.tokenizer, self.args)#去除尾实体不在vocab中的数据集
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer, self.args)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer, self.args)

        #结束
        # 5. DataLoader转换 (MindSpore GeneratorDataset)
        self.train_loader = GeneratorDataset(
            source=self.train_set,
            column_names=["sub", "obj"],
            shuffle=True,
            num_parallel_workers=4
        ).batch(batch_size=64, drop_remainder=True)
        
        self.dev_loader = GeneratorDataset(
            source=self.dev_set,
            column_names=["sub", "obj"],
        ).batch(batch_size=64)
        
        self.test_loader = GeneratorDataset(
            source=self.test_set,
            column_names=["sub", "obj"], # "labels"
        ).batch(batch_size=64)

        self.best_mrr = 0
        
        self.model = PTuneForLAMA(args, self.args.template)#将模板prompt进行了初始化编码

    def get_TREx_parameters(self):
        data_path_pre = "fact-retrieval/original/{}/".format(self.args.relation_id)
        data_path_post = ".jsonl"

        return data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        # 1. 设置模型为评估模式
        self.model.set_train(False)  # MindSpore的eval模式设置
        
        # 2. 选择数据集
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        
        # 3. 初始化指标
        hit1, hit3, hit10, loss, MRR = 0, 0, 0, 0, 0
        total_samples = 0
        
        for batch in loader.create_tuple_iterator():
            if len(batch) == 2:  # 假设x_hs, x_ts
                x_hs, x_ts = batch
            else:
                x_hs, x_ts, labels = batch
            
            x_hs = x_hs.asnumpy()
            x_ts = x_ts.asnumpy()
            token_ids = []
            label_ids = []
            for i in range(len(x_hs)):
                token = self.tokenizer_old.tokenize(str(x_hs[i]))
                token_id = self.tokenizer_old.convert_tokens_to_ids(token)
                token_ids.append(token_id)

                label_token_id = self.tokenizer.convert_tokens_to_ids(x_ts[i])
                label_ids.append([label_token_id])
            label_ids = ms.Tensor(label_ids, dtype=ms.int32) if not isinstance(label_ids, ms.Tensor) else label_ids
            current_loss, current_hit1, current_hit3, current_hit10, current_mrr = self.model(
                 token_ids, label_ids, evaluate_type, token_ids, epoch_idx
            )
            
            # 7. 指标累加 (使用MindSpore算子)
            hit1 += current_hit1.asnumpy() if isinstance(current_hit1, ms.Tensor) else current_hit1
            hit3 += current_hit3.asnumpy() if isinstance(current_hit3, ms.Tensor) else current_hit3
            hit10 += current_hit10.asnumpy() if isinstance(current_hit10, ms.Tensor) else current_hit10
            MRR += current_mrr.asnumpy() if isinstance(current_mrr, ms.Tensor) else current_mrr
            loss += current_loss.asnumpy() if isinstance(current_loss, ms.Tensor) else current_loss
            total_samples += x_hs.shape[0]  # 使用batch维度获取样本数
        
        # 8. 计算平均指标
        original_hit1 = hit1
        original_hit3 = hit3
        original_hit10 = hit10
        
        if total_samples > 0:
            hit1 /= total_samples
            hit3 /= total_samples
            hit10 /= total_samples
            div_loss = loss / total_samples
            avg_mrr = MRR / total_samples
        else:
            div_loss = 0
            avg_mrr = 0
        
        # # 9. 日志记录 (保持与原始代码一致)
        logger.info(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@1: {hit1}")
        logger.info(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@3: {hit3}")
        logger.info(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@10: {hit10}")
        logger.info(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  MRR: {avg_mrr}")
        
        print(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@1: {hit1}")
        print(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@3: {hit3}")
        print(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  Hit@10: {hit10}")
        print(f"{self.args.relation_id} {evaluate_type} Epoch {epoch_idx}  MRR: {avg_mrr}")
        
        return loss, original_hit1, original_hit3, original_hit10, total_samples, MRR


    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                            round(test_hit1 * 100, 4))
        
        # In MindSpore, we need to prepare the parameters to save
        param_dict = {}
        for name, param in self.model.prompt_encoder.parameters_and_names():
            param_dict[name] = param
            
        # Additional metadata to save
        metadata = {
            'dev_hit@1': Tensor(np.array(dev_hit1), mindspore.float32),
            'test_hit@1': Tensor(np.array(test_hit1), mindspore.float32),
            'test_size': Tensor(np.array(len(self.test_set)), mindspore.int32),
            'ckpt_name': ckpt_name,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'args': str(self.args)  # Convert args to string for saving
        }
        
        return {'params': param_dict, 'metadata': metadata, 'ckpt_name': ckpt_name}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        
        # Save parameters as MindSpore checkpoint
        save_checkpoint(best_ckpt['params'], join(path, ckpt_name))
        
        # Save metadata separately (MindSpore checkpoints don't natively support custom metadata)
        with open(join(path, ckpt_name + '.meta'), 'w') as f:
            f.write(str(best_ckpt['metadata']))
        
        logger.info("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

 
    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params_dict = {}
        
        # Prepare parameters for optimizer
        prompt_params = list(filter(lambda x: x.requires_grad, self.model.prompt_encoder.get_parameters()))
        if prompt_params:
            params_dict['prompt'] = {
                'params': prompt_params,
                'lr': self.args.lr
            }

        # 2. 添加模型参数（统一使用字典格式）
        if self.args.use_lm_finetune:
            # 获取模型参数并去重
            model_params = []
            seen_params = set()
            
            for param in self.model.model.get_parameters():
                if param.requires_grad and id(param) not in seen_params:
                    model_params.append(param)
                    seen_params.add(id(param))
            
            params_dict['model'] = {
                'params': model_params,
                'lr': ms.Tensor(5e-6)  # 更低的微调学习率
            }

        # 2. 转换为优化器需要的参数列表
        optimizer_params = list(params_dict.values())

        # 3. 创建优化器
        optimizer = nn.Adam(optimizer_params, learning_rate=self.args.lr, weight_decay=self.args.weight_decay)
        
        # Create learning rate scheduler
        lr_scheduler = nn.ExponentialDecayLR(self.args.lr, self.args.decay_rate, decay_steps=1, is_stair=False)
        
        # Define forward function
        def forward_fn(inputs, labels, token_ids):
            loss, logits= self.model(token_ids, labels, 'train', token_ids)

            return loss, logits
        
        # Define gradient function
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        for epoch_idx in range(200):
            # # Check early stopping
            if epoch_idx > -1:
                test_loss, test_hit1, test_hit3, test_hit10, len_dataset, mrr = self.evaluate(epoch_idx, 'Test')
                
                if epoch_idx > 0 and (mrr > self.best_mrr or self.args.only_evaluate):
                    self.best_hit1 = test_hit1
                    self.best_hit3 = test_hit3
                    self.best_hit10 = test_hit10
                    self.best_mrr = mrr
                    early_stop = 0
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        logger.info("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt, self.best_hit1, self.best_hit3, self.best_hit10, len_dataset, self.best_mrr
            
            if self.args.only_evaluate:
                break

            # Training loop
            hit1, hit3, hit10, num_of_samples = 0, 0, 0, 0
            tot_loss = 0
            
            for batch_idx, batch in tqdm(enumerate(self.train_loader.create_tuple_iterator())):
                self.model.set_train()
                if len(batch) == 2:  # 假设x_hs, x_ts
                    x_hs, x_ts = batch
                    labels = None
                else:
                    x_hs, x_ts, labels = batch

                token_ids = []
                label_ids = []
                for i in range(x_hs.shape[0]):
                    input_str = x_hs.asnumpy()[i] if isinstance(x_hs[i], ms.Tensor) else str(x_hs[i])
                    label_str = x_ts.asnumpy()[i] if isinstance(x_ts[i], ms.Tensor) else str(x_ts[i])
                    tokens = self.tokenizer_old.tokenize(input_str)
                    # label_tokens = self.tokenizer_old.tokenize(label_str)
                    token_id = self.tokenizer_old.convert_tokens_to_ids(tokens)
                    label_id = self.tokenizer.convert_tokens_to_ids(label_str)
                    token_ids.append([token_id])
                    label_ids.append([label_id])
                token_ids = ms.Tensor(token_ids, dtype=ms.int32) if not isinstance(token_ids, ms.Tensor) else token_ids
                label_ids = ms.Tensor(label_ids, dtype=ms.int32) if not isinstance(label_ids, ms.Tensor) else label_ids
                x_hs = label_ids
                (loss, logits), grads = grad_fn(x_hs, label_ids, token_ids)
           
                # Update parameters
                optimizer(grads)
                logger.info(f"{self.args.relation_id}  Epoch {epoch_idx} Loss: {loss}")
                print(f"{self.args.relation_id}  Epoch {epoch_idx} Loss: {loss}")
                
            # Update learning rate
            current_lr = lr_scheduler(ms.Tensor(epoch_idx, ms.float32))
            optimizer.learning_rate = current_lr
            
        return best_ckpt, self.best_hit1, self.best_hit3, self.best_hit10, len_dataset, self.best_mrr

def main(relation_id=None):

    relations = [
        'ConceptNet_relation_IsA', 
        'ConceptNet_relation_CapableOf', 
        'ConceptNet_relation_UsedFor', 
        'ConceptNet_relation_HasProperty', 
        'ConceptNet_relation_MadeOf', 
        'ConceptNet_relation_HasSubevent', 
        'ConceptNet_relation_AtLocation', 
        'ConceptNet_relation_PartOf', 
        'ConceptNet_relation_Causes', 
        'ConceptNet_relation_HasA', 
        'ConceptNet_relation_HasPrerequisite', 
        'ConceptNet_relation_ReceivesAction',
        'ConceptNet_relation_NotCapableOf', 
        'ConceptNet_relation_CausesDesire', 
        'ConceptNet_relation_Desires', 
        'ConceptNet_relation_MotivatedByGoal', 
        'ConceptNet_relation_NotIsA', 
        'ConceptNet_relation_HasFirstSubevent', 
        'ConceptNet_relation_NotHasProperty', 
        'ConceptNet_relation_CreatedBy', 
        'ConceptNet_relation_DefinedAs'
    ]

    templates = [
        "(4,2,0)","(4,4,0)","(4,2,0)","(4,2,0)","(4,4,4)",
        "(4,4,4)","(4,1,0)","(2,2,0)","(4,3,0)","(4,4,0)",
        "(5,5,0)","(2,4,0)","(4,4,0)","(4,4,0)","(4,4,0)",
        "(6,6,0)","(4,4,0)","(4,4,0)","(4,4,0)","(5,5,0)",
        "(4,4,0)"
    ]
    args = construct_generation_args()


    relation_best_prompt_templete_test_data = {}
    for index,relation in enumerate(relations):
        args.relation_id = relation
        if type(args.template) is not tuple:
            args.template = eval(args.template)
        assert type(args.template) is tuple

        logger.info(args.relation_id)
        logger.info(args.template)
        logger.info(args.model_name)
        logger.info(args)
        print(args.relation_id)
        print(args.template)
        print(args.model_name)
        print(args)
        trainer = Trainer(args)
        _,relation_best_hit1,relation_best_hit3,relation_best_hit10,len_dataset,relation_best_mrr = trainer.train()
        hit_data = []
        hit_data.append(relation_best_hit1)
        hit_data.append(relation_best_hit3)
        hit_data.append(relation_best_hit10)
        hit_data.append(relation_best_mrr)
        hit_data.append(len_dataset)
        relation_best_prompt_templete_test_data[relation] = hit_data
        logger.info(relation_best_prompt_templete_test_data)
        print(relation_best_prompt_templete_test_data)
        

    hit1 = 0
    hit3 = 0
    hit10 = 0
    dataset_len = 0
    MRR = 0
    for value in relation_best_prompt_templete_test_data.values():
        hit1 += value[0]
        hit3 += value[1]
        hit10 += value[2]
        MRR += value[3]
        dataset_len += value[4]


    logger.info('hit@1: {}'.format(hit1/dataset_len))
    logger.info('hit@3: {}'.format(hit3/dataset_len))
    logger.info('hit@10: {}'.format(hit10/dataset_len))
    logger.info('MRR: {}'.format(MRR/dataset_len))

    print('hit@1:',hit1/dataset_len)
    print('hit@3:',hit3/dataset_len)
    print('hit@10:',hit10/dataset_len)
    print('MRR:',MRR/dataset_len)

    


if __name__ == '__main__':
    main()
