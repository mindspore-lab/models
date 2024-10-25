import mindspore;mindspore.set_context(device_target="CPU")
import mindspore_lite as mslite
from mindspore import Tensor
from mindspore.ops import L2Normalize
from mindspore.ops import transpose
from mindnlp.transformers import AutoTokenizer

import numpy as np
import os

import argparse
parser = argparse.ArgumentParser(description='gen ckpt model')
parser.add_argument('--model-path', type=str, required=False, help='mindir path', default=r"./model/model.mindir")
parser.add_argument('--config-path', type=str, required=False, help='tokenizer config path', default=r"./bge-large-zh-v1.5")
args = parser.parse_args()

MODEL_PATH = args.model_path
TOKEN_PATH = args.config_path
OUTPUT_PATH = r"./output"

context = mslite.Context()
context.target = ["ascend"]
context.cpu.thread_num = 1
context.cpu.thread_affinity_mode=2



model = mslite.Model()
tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH)
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)

def GetInput(sentences):
    
    encoded_input = tokenizer(sentences, padding="max_length", truncation=True, return_tensors="ms", max_length=512)
    input_ids = encoded_input.input_ids
    attention_mask = encoded_input.attention_mask
    token_type_ids = encoded_input.token_type_ids
    
    inputs = model.get_inputs()

    inputs[0] = input_ids.asnumpy().astype(np.int32)
    inputs[1] = attention_mask.asnumpy().astype(np.int32)
    inputs[2] = token_type_ids.asnumpy().astype(np.int32)
    return inputs

#从文件读取字符串列表
def infer(model):
    sentences = ["样例数据-1", "样例数据-2"]
    inputs = GetInput(sentences)
    model_output = model.predict(inputs)
    data = model_output[0].get_data_to_numpy()
    sentence_embeddings = Tensor.from_numpy(data[:, 0])
    l2_normalize = L2Normalize(axis=1, epsilon=1e-12)
    final_outputs_1 = l2_normalize(sentence_embeddings)
    return final_outputs_1


#推理结果，可用于精度验证
final_outputs = infer(model)
print(final_outputs)