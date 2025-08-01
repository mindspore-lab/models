# retrieval_ms.py
# Evaluate retrieval performance on Arguana and Quora using MindSpore
import mindspore as ms
# [FIX] Set execution mode to PyNative to correctly handle the model's dynamic logic.
# This must be the first MindSpore call in your script.
ms.set_context(mode=ms.PYNATIVE_MODE)
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ops # Import Parameter
import numpy as np
from typing import List, Dict
from bpemb import BPEmb
from tqdm import trange
import sys
sys.path.append("/home/mingshi/Banyan-ms/")
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval



# Import the MindSpore version of the model
from models import Banyan

def pad_sequence(sequences: List[Tensor], batch_first: bool = True, padding_value: float = 0.0) -> Tensor:
    """Pads a list of variable length Tensors with padding_value."""
    max_len = max([s.shape[0] for s in sequences])
    out_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    
    out_tensor = ops.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        if batch_first:
            out_tensor[i, :length] = tensor
        else:
            out_tensor[:length, i] = tensor
    return out_tensor

# --- Main BEIR Evaluation Logic ---

class StrAE_DE_MindSpore:
    def __init__(self, model, **kwargs):
        self.model = model
        self.tokenizer = BPEmb(lang='en', vs=25000, dim=100)
        self.model.set_train(False) # Set model to evaluation mode

    def _encode_batch(self, texts: List[str]) -> Tensor:
        """Encodes a single batch of texts."""
        encoded = [Tensor(self.tokenizer.encode_ids(x), dtype=ms.int32) for x in texts]
        encoded_padded = pad_sequence(encoded, batch_first=True, padding_value=25000)
        
        # The model expects two identical inputs for the retrieval task
        model_out, _ = self.model(encoded_padded, encoded_padded)
        return model_out

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        query_embeddings = []
        for start_idx in trange(0, len(queries), batch_size, desc="Encoding Queries"):
            batch_texts = queries[start_idx:start_idx + batch_size]
            model_out = self._encode_batch(batch_texts)
            query_embeddings.append(model_out.asnumpy())
        
        return np.concatenate(query_embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Correctly handle corpus which can be a list of dicts or a dict of dicts
        if isinstance(corpus, dict):
            corpus = [corpus[cid] for cid in corpus]
            
        corpus_embeddings = []
        for start_idx in trange(0, len(corpus), batch_size, desc="Encoding Corpus"):
            batch_texts = [c.get('text', '') + " " + c.get('title', '') for c in corpus[start_idx:start_idx + batch_size]]
            model_out = self._encode_batch(batch_texts)
            corpus_embeddings.append(model_out.asnumpy())
            
        return np.concatenate(corpus_embeddings, axis=0)

# --- Script Execution ---
if __name__ == '__main__':
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    # These are the hyperparameters used in the paper
    model_ms = Banyan(vocab_size=25001, embedding_size=256, channels=128, r=0.1)

    if len(sys.argv) < 2:
        print("Error: Please provide the path to the checkpoint file.")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    print("Loading weights from checkpoint...")
    param_dict = ms.load_checkpoint(pth_path)
    ms.load_param_into_net(model_ms, param_dict)
    print("Weight loading complete.")

    # Wrap the MindSpore model for BEIR
    model_beir = DRES(StrAE_DE_MindSpore(model_ms), batch_size=64) # Reduced batch size for stability
    retriever = EvaluateRetrieval(model_beir, score_function="cos_sim")

    # --- Evaluate on Arguana ---
    data_path = "../data/arguana"
    print('Evaluating on Arguana dataset...')
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    results = retriever.retrieve(corpus, queries)
    print(retriever.evaluate(qrels, results, retriever.k_values))
    print('\n')

    # --- Evaluate on Quora ---
    data_path = "../data/quora"
    print('Evaluating on Quora dataset...')
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    results = retriever.retrieve(corpus, queries)
    print(retriever.evaluate(qrels, results, retriever.k_values))