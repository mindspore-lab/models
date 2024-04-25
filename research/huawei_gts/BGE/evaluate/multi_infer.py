import os
import argparse
import datasets
import numpy as np
import logging

import mindspore as ms
from mindspore import nn
from mindspore.communication import init, get_rank, get_group_size
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=False)
init()
ms.set_seed(1)
from mindnlp.transformers import AutoModel, AutoTokenizer

from FlagEmbedding import FlagModel

logger = logging.getLogger(__name__)
def split(data):
    rank_id = get_rank()
    rank_size = get_group_size()
    len_data = len(data)
    block_size = len_data // rank_size
    idx_st = 0
    head_data_num = len_data % rank_size
    if rank_id < head_data_num:
        block_size += 1
        idx_st = block_size * rank_id
    else:
        idx_st = (block_size + 1) * head_data_num + (rank_id - head_data_num) * block_size
    return data[idx_st: idx_st + block_size]

parser = argparse.ArgumentParser(description='Arguments for the encoder model')
parser.add_argument('--encoder', type=str, default='BAAI/bge-base-en-v1.5', help='The encoder name or path.')
parser.add_argument('--fp16', action='store_true', help='Use fp16 in inference?')
parser.add_argument('--add_instruction', action='store_true', help='Add query-side instruction?')
parser.add_argument('--max_passage_length', type=int, default=128, help='Max passage length.')
parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size.')
parser.add_argument('--corpus_embeddings', type=str, default='./corpus_embeddings', help='Output embeddings path')
args = parser.parse_args()


model = FlagModel(
    args.encoder, 
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
    use_fp16=args.fp16
)

#create_dataset
corpus = datasets.load_dataset("./msmarco-corpus", split="train")
sentences = corpus['content']
sentences = split(sentences)

#encode parallel
corpus_embeddings = model.encode_corpus(sentences, batch_size=args.batch_size, max_length=args.max_passage_length)

#save file
rank_id = get_rank()

directory = args.corpus_embeddings
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(args.corpus_embeddings + "/corpus_embeddings_rank{}.npy".format(rank_id),corpus_embeddings)