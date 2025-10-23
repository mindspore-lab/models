import os
import re
import pandas as pd
import pickle
import mindspore as ms
import numpy as np
import os
from sklearn.metrics import roc_auc_score, log_loss
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore import Model
from mindspore.dataset import GeneratorDataset
from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindspore import context

from dataset import load_csv_as_df, PLM4CTRDataset
from model import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, DeepCrossModel, DCN_LLaMA

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

tokenizer = AutoTokenizer.from_pretrained("llama2_7b")

llama_model = LlamaForCausalLM.from_pretrained("llama2_7b")

data_args = {
    "train_file": "data/ml-1m/proc_data/train.csv",
    "test_file": "data/ml-1m/proc_data/test.csv",
    "h5": "data/ml-1m/proc_data/ctr.h5",
    "h5_meta": "data/ml-1m/proc_data/ctr-meta.json",
}
train_df = load_csv_as_df(data_args["train_file"])
test_df = load_csv_as_df(data_args["test_file"])
total_datasets = {
    "train": PLM4CTRDataset.from_pandas(train_df[:len(train_df) // 9 * 8]),
    "valid": PLM4CTRDataset.from_pandas(train_df[len(train_df) // 9 * 8:]),
    "test": PLM4CTRDataset.from_pandas(test_df),
}
for split_name in ["train", "valid", "test"]:
    total_datasets[split_name]._post_setups(
        tokenizer=tokenizer,
        shuffle_fields=False,
        meta_data_dir=data_args["h5_meta"],
        h5_data_dir=data_args["h5"],
        mode=split_name,
        model_fusion="prefix",
        do_mlm_only=False,
    )


ctr_model = DeepCrossModel()
raw_net = DCN_LLaMA(ctr_model, llama_model)
loss_net = NetWithLossClass(raw_net)
train_net = TrainStepWrap(loss_net)
eval_net = PredictWithSigmoid(raw_net)

train_net.set_train()
train_net.network.network.llama.set_train(False)
model = Model(train_net)
train_dataset = GeneratorDataset(
    source=total_datasets["train"], 
    column_names=["batch_ids", "label", "token", "attention_mask"],
    shuffle=True
)
train_dataset = train_dataset.batch(128)

test_dataset = GeneratorDataset(
    source=total_datasets["test"], 
    column_names=["batch_ids", "label", "token", "attention_mask"]
)
test_dataset = test_dataset.batch(128)

def evaluate(model, dataset):
    batch_num = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()
    print('eval batch num', batch_num, 'batch size', batch_size)
    eval_data = dataset.create_tuple_iterator()
    begin_time = time.time()
    pred_list, label_list = [], []

    for _ in range(batch_num):
        data = next(eval_data)
        preds = model(*data)
        pred_list.extend(preds.asnumpy().tolist())
        label_list.extend(data[1].asnumpy().tolist())

    eval_time = time.time() - begin_time
    auc = roc_auc_score(y_true=label_list, y_score=pred_list)
    logloss = log_loss(y_true=label_list, y_pred=pred_list)
    return auc, logloss, eval_time

best_auc = 0
save_path = "./checkpoints"
_patience = 2
# training
for epoch in range(5):
    begin_time = time.time()
    model.train(1, train_dataset)
    train_time = time.time() - begin_time
    eval_auc, eval_ll, eval_time = evaluate(eval_net, test_dataset)
    print("EPOCH %d , train time: %.5f, test time: %.5f, auc: %.5f, "
            "logloss: %.5f" % (epoch, train_time, eval_time, eval_auc, eval_ll))

    if eval_auc > best_auc:
        best_auc = eval_auc
        ms.save_checkpoint(eval_net, save_path)
        print('model save in', save_path)
        patience = 0
    else:
        patience += 1
        if patience >= _patience:
            break