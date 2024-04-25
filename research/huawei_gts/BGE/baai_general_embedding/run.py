import os
import argparse

import mindspore as ms
from mindspore.communication import init
ms.set_context(mode=ms.GRAPH_MODE)
try:
    os.environ["MINDSPORE_HCCL_CONFIG_PATH"] = os.getenv("RANK_TABLE_FILE")
    rank_id = int(os.getenv("RANK_ID"))
    rank_size = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()
    ms.set_seed(1)
    ms.set_context(device_id=device_id)

    print(f"rank_id: {rank_id}")
    print(f"device_id: {device_id}")
    print(f"rank_size: {rank_size}")

    print("distribute training...")
except TypeError:
    print("standalone training...")

from mindspore import nn
from mindspore.train import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

from mindnlp.transformers import AutoTokenizer
from modeling import BiEncoderModel
from data import TrainDatasetForEmbedding, process_data
from arguments import DataArguments
from lr_schedule import LinearWithWarmUpLR


def main(args):
    use_parallel = args.use_parallel
    model_name = args.model_name_or_path
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    train_data = args.train_data
    train_group_size = args.train_group_size
    query_max_len = args.query_max_len
    passage_max_len = args.passage_max_len
    save_steps = args.save_steps
    output_dir = args.output_dir
    temperature = args.temperature
    query_instruction_for_retrieval = args.query_instruction_for_retrieval

    model = BiEncoderModel(model_name, temperature=temperature)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_arguments = DataArguments()
    data_arguments.train_data = train_data
    data_arguments.train_group_size = train_group_size
    data_arguments.query_max_len = query_max_len
    data_arguments.passage_max_len = passage_max_len
    data_arguments.query_instruction_for_retrieval = query_instruction_for_retrieval

    train_dataset = TrainDatasetForEmbedding(args=data_arguments, tokenizer=tokenizer)
    dataset = process_data(train_dataset, batch_size=batch_size, use_parallel=use_parallel)

    total_steps = len(train_dataset) // batch_size * epoch

    lr_scheduler = LinearWithWarmUpLR(learning_rate=lr, warmup_steps=1, total_steps=total_steps)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_scheduler, beta1=0.9,
                                   beta2=0.999, eps=1e-8, weight_decay=0.01)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_steps, keep_checkpoint_max=1)
    ckpt_callback = ModelCheckpoint(directory=output_dir, config=ckpt_config)

    model = ms.Model(model, optimizer=optimizer)
    model.train(epoch, dataset, callbacks=[LossMonitor(), TimeMonitor(), ckpt_callback])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_parallel", action="store_true", help="whether use parallel")
    parser.add_argument("--model_name_or_path", default='./bge-large-zh-v1.5', type=str, help="model path")
    parser.add_argument("--epoch", default=1, type=int, help="num of epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--train_data", default='./toy_finetune_data.jsonl', type=str, help="train data path")
    parser.add_argument("--train_group_size", default=2, type=int, help="train group size, num of pos + neg samples")
    parser.add_argument("--query_max_len", default=256, type=int, help="query max length")
    parser.add_argument("--passage_max_len", default=256, type=int, help="passage max length")
    parser.add_argument("--save_steps", default=1000, type=int, help="checkpoint save steps")
    parser.add_argument("--output_dir", default='./checkpoint', type=str, help="checkpoint save path")
    parser.add_argument("--query_instruction_for_retrieval", default='', type=str, help="instruction for query")
    parser.add_argument("--temperature", default=0.02, type=float, help="influence the distribution of similarity scores")

    args, _ = parser.parse_known_args()
    print(args)
    main(args)
