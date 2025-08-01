# banyan_ms.py
import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

from funcs import Compose
from model_utils import create_index, get_sims, reduce_frontier, get_complete

class SimpleGraph:
    """
    A lightweight class to simulate DGL graph functionalities needed for inference.
    It stores node embeddings and tracks composition history to enable the 'entangled tree'.
    """
    def __init__(self, initial_node_embeddings):
        self.node_data = initial_node_embeddings
        self.composition_history = {}
        self.num_nodes = initial_node_embeddings.shape[0]

    def get_node_embeddings(self, indices):
        return self.node_data[indices]

    def add_nodes_and_edges(self, new_parent_embeddings, child_pairs):
        """
        Adds new parent nodes and records the edges (composition history).
        """
        num_new_nodes = new_parent_embeddings.shape[0]
        if num_new_nodes == 0:
            return []

        new_node_indices = ops.arange(self.num_nodes, self.num_nodes + num_new_nodes)
        
        self.node_data = ops.cat((self.node_data, new_parent_embeddings), axis=0)

        for i, pair in enumerate(child_pairs):
            # pair is a tuple (child1_idx, child2_idx)
            self.composition_history[pair] = new_node_indices[i].asnumpy().item()
        
        self.num_nodes += num_new_nodes
        return new_node_indices


class Banyan(nn.Cell):
    """
    MindSpore implementation of the Banyan model for inference.
    DGL is replaced by the SimpleGraph class.
    """
    def __init__(self, vocab_size, embedding_size, channels, r):
        super(Banyan, self).__init__()
        self.E = embedding_size
        self.c = channels
        self.e = int(self.E / self.c)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size - 1)
        if r != 0.0:
            self.embedding.embedding_table.set_data(
                ms.common.initializer.initializer(ms.common.initializer.Uniform(r), 
                                                   self.embedding.embedding_table.shape, 
                                                   self.embedding.embedding_table.dtype)
            )
        
        pad_embedding = ops.ones(embedding_size) * -1e9
        self.embedding.embedding_table.data[vocab_size - 1] = pad_embedding

        self.comp_fn = Compose(self.E, self.c)
        self.vocab_size = vocab_size
        
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Dense(self.E, self.vocab_size - 1)

    def update_and_compose(self, graph, retrieval, index):
        """
        A combined and simplified version of the original `update_graph`.
        This function finds which pairs need composition, composes them,
        updates the graph, and returns the new parent indices.
        """
        range_tensor = ops.arange(index.shape[0]).repeat_interleave(2)
        
        src_pairs_tensor = index[range_tensor, retrieval].view(-1, 2)
        
        new_compositions_mask = []
        existing_parent_indices = {} # (row_in_batch, original_pair_tuple) -> existing_parent_idx
        
        src_pairs_list = [tuple(p.asnumpy()) for p in src_pairs_tensor]

        for i, pair in enumerate(src_pairs_list):
            if pair in graph.composition_history:
                existing_parent_indices[(i // 2, pair)] = graph.composition_history[pair]
                new_compositions_mask.append(False)
            else:
                new_compositions_mask.append(True)

        new_compositions_mask = Tensor(new_compositions_mask, ms.bool_)
        
        unique_new_pairs_tensor, unique_indices = ops.unique(src_pairs_tensor[new_compositions_mask], return_inverse=True)
        
        if unique_new_pairs_tensor.shape[0] > 0:
            children_embeds = graph.get_node_embeddings(unique_new_pairs_tensor)
            parent_embeds = self.comp_fn(children_embeds.view(-1, 2, self.c, self.e))
            new_parent_node_indices = graph.add_nodes_and_edges(parent_embeds, [tuple(p.asnumpy()) for p in unique_new_pairs_tensor])
            new_parents_for_batch = new_parent_node_indices[unique_indices]
        
        update_indices = ops.zeros(src_pairs_tensor.shape[0], dtype=ms.int32)
        if new_compositions_mask.any():
            update_indices[new_compositions_mask] = new_parents_for_batch

        for (row_idx, pair), parent_idx in existing_parent_indices.items():

            for i, p in enumerate(src_pairs_list):
                if i // 2 == row_idx and p == pair:
                    update_indices[i] = parent_idx
        

        index[range_tensor, retrieval] = update_indices.view(-1)
        return graph, index


    def construct_compose(self, seqs, roots=False):
        """
        The core composition logic for inference, rewritten for MindSpore.
        """
        range_tensor = Tensor(range(seqs.shape[0]), dtype=ms.int64)
        index, tokens, _ = create_index(seqs)
        
        initial_embeds = self.dropout(self.embedding(tokens))
        g = SimpleGraph(initial_embeds)

        while index.shape[1] > 1:
            node_embeds = g.get_node_embeddings(index)
            max_sim, retrieval = get_sims(node_embeds)
            completion_mask = get_complete(index)
            
            g, index = self.update_and_compose(g, retrieval[completion_mask.repeat_interleave(2)], index)
            
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)
        
        if roots:
            root_indices = index.flatten()
            return g.get_node_embeddings(root_indices)

        raise NotImplementedError("Inference mode only supports returning root embeddings.")

    def construct(self, seqs1, seqs2):
        """
        Main entry point for inference, mimicking the STS evaluation path.
        """
        r1 = self.construct_compose(seqs1, roots=True)
        r2 = self.construct_compose(seqs2, roots=True)
        return r1, r2