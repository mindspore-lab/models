import mindspore;mindspore.set_context(device_target="CPU")
from mindnlp.transformers import MSBertModel

import argparse
parser = argparse.ArgumentParser(description='gen ckpt model')
parser.add_argument('--model-path', type=str, required=False, help='huggingface model path', default=r"./bge-large-zh-v1.5")
args = parser.parse_args()

MODEL_PATH = args.model_path
CKPT_PATH = r"./model.ckpt"

model = MSBertModel.from_pretrained(MODEL_PATH)
mindspore.save_checkpoint(model, CKPT_PATH)