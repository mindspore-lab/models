import mindspore;mindspore.set_context(device_target="CPU")
from mindnlp.transformers import AutoTokenizer
from mindnlp.transformers import MSBertModel
from mindnlp.transformers.models import BertConfig

import bge_injection
import os
import json

import argparse
parser = argparse.ArgumentParser(description='gen ckpt model')
parser.add_argument('--model-path', type=str, required=False, help='huggingface model path', default=r"./bge-large-zh-v1.5")
args = parser.parse_args()

#MODEL_PATH 存放ckpt模型的路径
#TOKEN_PATH 存放词表与tokenizer配置文件的路径
model_path = r"./model/model.ckpt"
model_config_path = args.model_path
OUTPUT_MINDIR_PATH = r"./model/model.mindir"

#输入ckpt & config.json 输出model
def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig()
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config
def from_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
    return from_dict(json.loads(text))
def load_model(ckpt_path, model_config_folder_path):
    
    config_path = os.path.join(model_config_folder_path, 'config.json')
    
    bert_config = from_json_file(config_path)
    model = MSBertModel(bert_config)

    parameter_dict = mindspore.load_checkpoint(ckpt_path)

    param_not_load, ckpt_not_load = mindspore.load_param_into_net(model, parameter_dict)
    print("param_not_load: \n", param_not_load)
    print("ckpt_not_load: \n", ckpt_not_load)
    return model


model = load_model(model_path, model_config_path)

sentences_1 = ["样例数据-1", "样例数据-2"]
tokenizer = AutoTokenizer.from_pretrained(model_config_path)
encoded_input = tokenizer(sentences_1, padding="max_length", truncation=True, return_tensors="ms", max_length=512)
model_output = model(**encoded_input)

mindspore.export(model, encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids,  file_name=OUTPUT_MINDIR_PATH, file_format="MINDIR")