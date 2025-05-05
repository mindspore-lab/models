import argparse
from utils import str2bool
from mindnlp.transformers import AutoTokenizer
from models.Qwen.modeling_qwen_moe import Qwen2MoeForCausalLM
import mindspore as ms
from mindnlp.core import set_default_dtype

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B", help="The model name.")
    parser.add_argument("--path", type=str, default="model_weights", help="The path to the model weights.")
    parser.add_argument("--early_stopping", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--memory_budget", type=int, default=0, help="GB")
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--overlap", type=str2bool, nargs='?', const=True, default=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    set_default_dtype(ms.float16)
    import mindspore.context as context
    context.set_context(pynative_synchronize=True)

    model_name = args.model
    if model_name == "Qwen/Qwen1.5-MoE-A2.7B":
        tokenizer = AutoTokenizer.from_pretrained("model_weights/qwen1.5-moe-a2.7b/tokenizer")
        model = Qwen2MoeForCausalLM(args)
    
    model.eval()

    input_prompt = "Hey, are you conscious? Can you talk to me?"
    input_tokenizer = tokenizer(input_prompt, return_tensors="ms")
    input_ids = input_tokenizer.input_ids
    attention_mask = input_tokenizer.attention_mask

    (output_ids, prefill_time) = model.generate(input_ids, attention_mask=attention_mask, expriment_mode="decoding")

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)
    # outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(outputs)