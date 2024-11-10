import json
import argparse
import mindspore
from mindformers import LlamaForCausalLM, LlamaTokenizer, MindFormerConfig, LlamaConfig, init_context
from tqdm import trange
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


def get_prompt(pair):
    B_INST, E_INST = "<s>[INST]", "[/INST] "
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + pair["input"] + E_INST + pair["output"]
    return prompt_template



def preprocess_logits_for_metrics(logits, labels):
    """
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    labels_index = ops.nonzero(ops.bitwise_or(labels == 3869, labels == 1939))

    gold = ops.select(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 
                      Tensor(0, mindspore.int32), Tensor(1, mindspore.int32))
    
    labels_index[:, 1] = labels_index[:, 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = ops.Softmax(axis=-1)(logits)

    return prob[:, 1], gold


def compute_metrics(prob, gold):
    auc = roc_auc_score(gold, prob)
    ll = log_loss(gold, prob)
    acc = accuracy_score(gold, prob > 0.5)
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }
    

def infer(args):
    dataset = json.load(open(args.data_dir))
    dataset = [get_prompt(pair) for pair in dataset]

    llama_config = MindFormerConfig("run_llama2_7b_910b.yaml")

    init_context(use_parallel=llama_config.use_parallel,
                    context_config=llama_config.context,
                    parallel_config=llama_config.parallel)

    model_config = LlamaConfig(llama_config.model.model_config)
    model_config.use_past = True
    model_config.seq_length = 2048
    model_config.checkpoint_name_or_path = args.ckpt_dir

    tokenizer = LlamaTokenizer.from_pretrained("llama2_7b")
    model = LlamaForCausalLM(model_config)
    model.set_train(False)


    prob, gold = [], []

    for i in trange(0, len(dataset), args.eval_batch_size):
        cur_batch = dataset[i:i + args.eval_batch_size]
        inputs = tokenizer(cur_batch)
        for k in inputs:
            inputs[k] = inputs[k].cuda()
        logits, _, _ = model.construct(**inputs)
        cur_prob, cur_labels = preprocess_logits_for_metrics(logits, inputs['input_ids'])
        prob.append(cur_prob)
        gold.append(cur_labels)
    
    prob = mnp.concatenate(prob, axis=0)
    gold = mnp.concatenate(gold, axis=0)
    metrics = compute_metrics(prob, gold)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str)
    args = parser.parse_args()
    infer(args)