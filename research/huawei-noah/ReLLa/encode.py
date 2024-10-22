import json
import argparse
import mindspore.numpy as mnp
from mindformers import LlamaModel, LlamaTokenizer, MindFormerConfig, LlamaConfig, init_context


def encode(args):
    dataset = json.load(open(args.item_dir))

    llama_config = MindFormerConfig("run_llama2_7b_910b.yaml")

    init_context(use_parallel=llama_config.use_parallel,
                    context_config=llama_config.context,
                    parallel_config=llama_config.parallel)

    model_config = LlamaConfig(llama_config.model.model_config)
    model_config.seq_length = 2048
    model_config.checkpoint_name_or_path = args.ckpt_dir

    tokenizer = LlamaTokenizer.from_pretrained("llama2_7b")
    model = LlamaModel(model_config)
    model.set_train(False)

    embeddings = []
    for data in dataset:
        inputs = tokenizer([data])
        for k in inputs:
            inputs[k] = inputs[k].cuda()
        hidden_embed = model.construct(**inputs)
        embeddings.append(hidden_embed[0].mean(dim=0).asnumpy())
    embeddings = mnp.stack(embeddings, axis=0)
    mnp.save(args.embed_dir, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_dir", type=str, help="Path to the item descriptions. Should be a .json file.")
    parser.add_argument("--embed_dir", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    args = parser.parse_args()

    encode(args)