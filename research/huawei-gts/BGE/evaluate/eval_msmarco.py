import faiss
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
import argparse
from FlagEmbedding import FlagModel
import logging
import os
logger = logging.getLogger(__name__)

import mindspore as ms
def read_from_folder(folder_path):
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.npy') == True and f.endswith('gpu.npy') == False]
    file_names.sort()
    first_file = np.load(os.path.join(folder_path, file_names[0]))
    sum_size = 0
    for i, file_name in enumerate(file_names):
        print("step:",i)
        file_path = os.path.join(folder_path, file_name)
        array = np.load(file_path)
        sum_size += array.shape[0]
    concatenated_array = np.zeros((sum_size, *first_file.shape[1:]))
    cur_size = 0
    for i, file_name in enumerate(file_names):
        print("step:",i)
        file_path = os.path.join(folder_path, file_name)
        array = np.load(file_path)
        concatenated_array[cur_size:cur_size + array.shape[0]] = array
        cur_size += array.shape[0]
    return concatenated_array
def index(model: FlagModel, corpus: datasets.Dataset, corpus_embeddings_path: str, batch_size: int = 256, max_length: int=512, index_factory: str = "Flat"):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    corpus_embeddings = read_from_folder(corpus_embeddings_path)
    dim = corpus_embeddings.shape[-1]
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(model: FlagModel, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["query"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
    
def evaluate(preds, labels, cutoffs=[1,10,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    return metrics

def main():
    parser = argparse.ArgumentParser(description='Arguments for the encoder model')
    parser.add_argument('--encoder', type=str, default='BAAI/bge-base-en-v1.5', help='The encoder name or path.')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 in inference?')
    parser.add_argument('--add_instruction', action='store_true', help='Add query-side instruction?')
    parser.add_argument('--max_query_length', type=int, default=32, help='Max query length.')
    parser.add_argument('--max_passage_length', type=int, default=128, help='Max passage length.')
    parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size.')
    parser.add_argument('--index_factory', type=str, default='Flat', help='Faiss index factory.')
    parser.add_argument('--k', type=int, default=100, help='How many neighbors to retrieve?')
    parser.add_argument('--corpus_embeddings', type=str, default='./corpus_embeddings', help='Output embeddings path')
    
    ms.set_context(device_target="Ascend")
    args = parser.parse_args()
    eval_data = datasets.load_dataset("./msmarco", split="dev")
    corpus = datasets.load_dataset("./msmarco-corpus", split="train")

    model = FlagModel(
        args.encoder, 
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
        use_fp16=args.fp16
    )
    
    faiss_index = index(
        model=model, 
        corpus=corpus, 
        corpus_embeddings_path=args.corpus_embeddings,
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["content"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])

    metrics = evaluate(retrieval_results, ground_truths)

    print(metrics)


if __name__ == "__main__":
    main()
