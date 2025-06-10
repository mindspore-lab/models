import os
import sys
import csv
import torch
import string
import argparse
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
from mindspore.common import initializer as init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, Model
import mindformers
from mindformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from mindformers import GenerationConfig
from mindformers import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.models.llama import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint



os.environ['NUMEXPR_MAX_THREADS'] = '32'
import time

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():  device = "mps"
except:
    pass


def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0

    while up - bottom > 1:
        index = int((up + bottom) / 2)

        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index

    if up - bottom == 1:

        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up

    return index


def main(name: str = "",
         number_stego: int = 100,
         count: int = 1,
         map: str = "",
         config_path: str = "./predict_llama2_7b.yaml",
         use_parallel: bool = "",
         load_checkpoint: str = "./LLM/LLaMA2-7B"
         ):
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]
    batch_size = len(inputs)

    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = use_parallel
    device_num = os.getenv('MS_WORKER_NUM')
    logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
    config.parallel_config.model_parallel = int(device_num) if device_num else 1
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.batch_size = batch_size
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None
    model_name = config.trainer.model_name

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    def DAIRstega_embedding(
            instruction,
            input=None,
            temperature=0.7,
            top_p=0.75,
            top_k=100,
            num_beams=1,
            max_new_tokens=512,
            stream_output=False,
            count=count,
            PRECISION=48,
            map=map,  # sqrt / sqrt3 / linear
            **kwargs
    ):

        if config.load_checkpoint:
            logger.info("----------------Transform and load checkpoint----------------")
            seq_length = config.model.model_config.seq_length
            # set auto transform ckpt
            if os.path.isdir(config.load_checkpoint) or config.use_parallel:
                config.auto_trans_ckpt = True
            else:
                config.auto_trans_ckpt = False
            input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

        input_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")


        generation_config = GenerationConfig(temperature=temperature,
                                             top_p=top_p,
                                             top_k=top_k,
                                             num_beams=num_beams,
                                             early_stopping=True,
                                             do_sample=False,
                                             **kwargs
                                             )

        # GenerationConfig
        generate_params = {"input_ids": input_ids,
                           "generation_config": generation_config,
                           "return_dict_in_generate": True,
                           "output_scores": True,
                           "max_new_tokens": max_new_tokens}

        # -------------- Without streaming --------------
        with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream
        # bit_stream = '0001010000100100000101010001010010010'
        bit_index = int(torch.randint(0, high=1000, size=(1,)))  # or 0

        with torch.no_grad():
            start = time.time()
            stega_text, stega_bits = [], ''

            # PRECISION = 16
            # max_val = 2 ** PRECISION  # num of intervals
            # cur_interval = [0, max_val]  # bottom inclusive, top exclusive
            for i in range(max_new_tokens - 1):
                if '</s>' in stega_text:
                    break

                generation_output1 = network(input_ids)
                log_prob = generation_output1.logits
                prob = torch.softmax(log_prob, dim=-1)[:, -1, :].reshape(-1)
                # prob[1] = 0  # set unk to zero
                prob = prob / prob.sum()
                prob, indices = prob.sort(descending=True)
                # start recursion
                bit_tmp = 0
                PRECISION = PRECISION
                max_val = 2 ** PRECISION  # num of intervals
                cur_interval = [0, max_val]  # bottom inclusive, top exclusive
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range

                if prob[-1] < cur_threshold:
                    k = max(2, (prob < cur_threshold).nonzero()[0].item())
                    prob = prob[:k]
                    indices = indices[:k]

                prob = prob[:top_k]
                indices = indices[:top_k]

                if map == "sqrt":
                    prob = torch.round(torch.sqrt(prob), decimals=4)
                elif map == "sqrt3":
                    prob = torch.pow(prob, 1 / 3)
                elif map == "sqrt4":
                    prob = torch.pow(prob, 1 / 4)
                prob = prob / prob.sum()
                prob = prob.double()
                prob *= cur_int_range
                prob = prob.round().long()

                cum_probs = prob.cumsum(0)
                overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])

                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                cum_probs += cur_int_range - cum_probs[-1]

                cum_probs += cur_interval[0]

                message_bits = bit_stream[bit_index: bit_index + PRECISION]
                message_bits = [int(_) for _ in message_bits]
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, PRECISION)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, PRECISION)))

                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                prev = indices[selection].view(1, 1)

                gen = int(prev)
                input_ids = torch.cat([input_ids, torch.LongTensor([[gen]]).to(device)], dim=1).to(device)
                stega_bits += bit_stream[bit_index:bit_index + num_bits_encoded]
                bit_index += num_bits_encoded

                # if gen == 29889:
                #     break
                if gen == 29889:
                    # print(f"{gen},{tokenizer.decode(gen)}")
                    count -= 1
                    if 0 == count:
                        break

        # scores = generation_output.scores
        # s = generation_output.sequences[0]
        end = time.time()
        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # yield prompter.get_response(output)
        costTime = end-start
        print(stega_bits)
        return output, stega_bits, costTime



    def count_words(s):
        for char in string.punctuation:
            s = s.replace(char, '')
        words = s.split()
        return len(words)

    Instruction = []
    iter = number_stego
    last_part = os.path.basename(name)
    prefix, style = last_part.rsplit("_", 1)
    if prefix == "P":
        Instruction = [
            f"Generate a text in the style of {style}.",
            f"You are a writer who knows {style} very well. Generate a text in the style of {style}.",
        ]

    else:
        last_p = last_part.split("T_")[1]
        Instruction = [
            "Generate the content about the " + last_p + " theme."
        ]

    csv_file = name + ".csv"
    row = ["Instruction", "Response", "stega_bits", "costTime(s)", "words", "bits", "Embedding rate"]

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    for instr in Instruction:

        for j in range(iter):
            print(last_part, ":", j, "/", iter)
            output, stega_bits, costTime = DAIRstega_embedding(instruction=instr,
                                                               count=count,
                                                               map=map
                                                               )

            output = output.split("Response:\n", 1)[1]
            words = count_words(output)
            chars = len(stega_bits)
            row = [instr,
                   output,
                   stega_bits,
                   costTime,
                   words,
                   chars,
                   chars/words
                   ]

            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--use_parallel', action='store_true', default=False,
                        help='if run model prediction in parallel mode.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')

    args = parser.parse_args()
    main(
        args.config_path,
        args.use_parallel,
        args.load_checkpoint
    )
