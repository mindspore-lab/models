# models_ms.py
# Main func defining the Banyan Self-Structuring AutoEncoder for MindSpore
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Uniform, initializer
import numpy as np

# Import MindSpore-compatible utility functions
from model_utils import create_index, get_sims, reduce_frontier, get_complete
from funcs import Compose

# --- Helper class to simulate DGL Graph for inference ---
class StaticGraph:
    """A lightweight container to hold node data, replacing DGL for inference."""
    def __init__(self):
        self.ndata = {}
        self.node_count = 0
    
    def add_nodes(self, num, data=None):
        self.node_count += num
        if data:
            for key, value in data.items():
                self.ndata[key] = value

# --- Helper for initialization ---
def uniform_(tensor: Parameter, a: float = 0.0, b: float = 1.0) -> None:
    """
    Fills the input Parameter with values drawn from the uniform distribution U(a, b).
    
    [FIXED] This function has been corrected to align with MindSpore's API.
    MindSpore's Uniform initializer only supports U(0, scale), so we generate
    a tensor in the range [0, b-a] and then shift it by 'a'.

    tensor.set_data(ms.common.initializer.initializer(Uniform(scale=b-a, loc=a), tensor.shape, tensor.dtype))
    """
    # 1. Create an initializer for U(0, b-a)
    temp_init = Uniform(scale=b - a)
    # 2. Generate the tensor with this initializer
    data = initializer(temp_init, tensor.shape, tensor.dtype)
    # 3. Add the offset 'a' to shift the distribution to [a, b]
    data = data + a
    # 4. Set the new data to the parameter
    tensor.set_data(data)


class Banyan(nn.Cell):
    def __init__(self, vocab_size, embedding_size, channels, r):
        super(Banyan, self).__init__()
        self.E = embedding_size
        self.c = channels
        self.e = int(self.E / self.c)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size - 1)
        
        if r != 0.0:
            uniform_(self.embedding.embedding_table, -r, r)
        
        embedding_data = self.embedding.embedding_table.asnumpy()
        embedding_data[vocab_size - 1] = -np.inf
        self.embedding.embedding_table.set_data(Tensor(embedding_data))
        
        self.comp_fn = Compose(self.E, self.c)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Dense(self.E, self.vocab_size - 1)
        
    def compose(self, seqs, roots=False):
        """
        This function is heavily refactored to remove DGL dependency.
        Instead of creating a graph and updating it, we manage a node embedding tensor directly.
        """
        range_tensor = ops.arange(seqs.shape[0], dtype=ms.int32)
        index, tokens, leaf_inds = create_index(seqs, self.vocab_size)
        
        # Node features are stored in a tensor, not a graph object
        node_features = self.dropout(self.embedding(tokens))
        num_new_nodes = 0
        '''
        # Use a static for-loop instead of a dynamic while-loop.
        # The loop runs for the maximum possible number of compositions.
        initial_seq_len = index.shape[1]
        for _ in range(initial_seq_len - 1):
            # For batches with shorter sequences, we can break early.
            if index.shape[1] <= 1:
                break
        '''
        while index.shape[1] > 1:
 
            max_sim, retrieval = get_sims(ops.stop_gradient(node_features), index)
            completion_mask = get_complete(index)
            
            # --- Graph Update Simulation ---
            # Instead of updating a graph, we compute new node features and append them.
            if completion_mask.any():
                # 1. Get pairs to be composed
                range_sub_tensor = range_tensor[completion_mask].repeat_interleave(2)
                '''
                retrieval_sub = retrieval[completion_mask.repeat_interleave(2)]
                src_indices = index[range_sub_tensor, retrieval_sub].view(-1, 2)
                '''
                repeated_mask = ops.cast(completion_mask, ms.int8).repeat_interleave(2)
                repeated_mask = ops.cast(repeated_mask, ms.bool_)
                retrieval_sub = retrieval[repeated_mask]
                
                src_indices = index[range_sub_tensor, retrieval_sub].view(-1, 2)
                
                # 2. Compose them to get new node features
                new_node_feats = self.comp_fn(node_features[src_indices].view(-1, 2, self.c, self.e))
                
                # 3. Append new features to the main node feature tensor
                node_features = ops.cat((node_features, new_node_feats), axis=0)
                
                # 4. Update the index tensor to point to these new nodes
                new_node_indices = tokens.shape[0] + num_new_nodes + ops.arange(new_node_feats.shape[0], dtype=ms.int32)
                num_new_nodes += new_node_feats.shape[0]
                
                # Update the right child's position in the index with the new parent index
                index[range_sub_tensor[1::2], retrieval_sub[1::2]] = new_node_indices
                
            # Reduce the frontiers
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)
        
        # The final root embeddings are in the node_features tensor at the indices specified by the final index tensor
        root_embeddings = node_features[index.flatten()]
        
        if roots:
            return root_embeddings

        # For inference, we only need the final root embeddings.
        # The original code returned a graph and traversal order for decomposition,
        # which is not needed for this retrieval task.
        return root_embeddings, tokens

    def compose_words(self, word_sequence):
        """
        [NEW] MindSpore implementation of the greedy composition for single word sequences.
        This method is essential for word-level intrinsic evaluation.
        """
        # Embed the initial sequence of word IDs
        word_sequence = self.embedding(word_sequence)
        
        # Iteratively compose until only one root vector remains
        while word_sequence.shape[0] != 1:
          # [FIX] Hardened against Ascend TBE compiler failure for ArgMaxV2.
            # The TBE kernel for argmax fails when the input tensor has only one element.
            # This occurs in the last loop iteration when word_sequence has 2 elements.
            # We bypass the argmax call entirely for this deterministic edge case.
            if word_sequence.shape[0] == 2:
                # If only two vectors are left, the index of the pair is trivially 0.
                max_indices = Tensor(0, dtype=ms.int64)
            else:
                seq_float = ops.cast(word_sequence, ms.float32)
                cosines = ops.cosine_similarity(seq_float[:-1], seq_float[1:], dim=1)
                # Use default argmax for all other valid cases.
                max_indices = ops.argmax(cosines)
            
            # [FIX] Corrected and simplified the creation of the retrieval tensor.
            # The ops.transpose call was erroneous as it cannot be applied to a 1D tensor and has been removed.
            # ops.stack is a more direct way to create the desired [index, index + 1] tensor.
            retrieval = ops.cast(ops.stack([max_indices, max_indices + 1]), ms.int32)

            
            # Select the pair and compose them into a parent vector
            batch_selected = word_sequence[retrieval].view(2, self.E)
            parent = self.comp_fn(batch_selected.view(2, self.c, self.e), words=True)
            
            # [FIX] Replaced the faulty tensor_scatter_elements with direct indexed assignment.
            # This is the correct, idiomatic way to perform this operation in PyNative mode
            # and directly mirrors the original PyTorch logic.
            update_idx = ops.cast(max_indices + 1, ms.int32)
            word_sequence[update_idx] = parent
            
            # 2. Create a mask to remove the first vector of the pair.
            batch_remaining_mask = ops.ones(word_sequence.shape[0], ms.bool_)
            batch_remaining_mask[ops.cast(max_indices, ms.int32)] = False
            
            # 3. Apply the mask to shorten the sequence
            word_sequence = word_sequence[batch_remaining_mask].view(-1, self.E)
            
        return word_sequence.squeeze()

    def construct(self, seqs, seqs2=None, words=False):
        # The original `forward` is now `construct`.
        # The logic for `words` is not used in retrieval.py, so it's kept for completeness but not essential.
        if words:
            # This path is now fully functional for word-level evaluation.
            return self.compose_words(seqs)

        # For STS (as in the retrieval task)
        if seqs2 is not None: 
            r1 = self.compose(seqs, roots=True)
            r2 = self.compose(seqs2, roots=True)
            return r1, r2

        # This part of the original code was for auto-encoding, not retrieval.
        # The retrieval task calls the model with two arguments (seqs, seqs2).
        # We return the root embedding directly.
        root_embeddings, _ = self.compose(seqs)
        # The projection head `self.out` is for reconstruction loss, not used in retrieval embedding.
        return root_embeddings, _