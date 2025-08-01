# model_utils_ms.py
# Helper functions for mapping sequences to indices and reducing frontiers
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
'''
def create_index(seqs: Tensor):
    """
    Creates the index tensor for updating the adjacency matrix -> maps each token to a unique index matching its node.
    """
    unique_values = ops.unique(seqs.flatten())[0]
    # In MindSpore, it's often easier to work with hash maps for lookups
    # or to use array indexing if the vocab size is manageable.
    lookup = ops.zeros(unique_values.max() + 1, dtype=ms.int32)
    lookup[unique_values] = ops.arange(len(unique_values), dtype=ms.int32)
    # set the final value to -1 to represent padding
    lookup[-1] = -1
    index = lookup[seqs]
    # padding is set to -1 so we filter it out
    leaf_inds = ops.unique(index.flatten())[0]
    leaf_inds = leaf_inds[leaf_inds != -1]
    
    # remove the padding value, :-1 because pad = 25k for tokens
    unique_values = unique_values[:-1] 
    return index, unique_values, leaf_inds
'''
def create_index(seqs: Tensor, vocab_size: int):
    """
    [FIXED] Creates a robust index tensor for updating the adjacency matrix.
    This version correctly and explicitly handles the padding ID.
    """
    # The padding token ID is the last one in the vocabulary.
    padding_id = vocab_size - 1
    
    # Get all unique token IDs present in the batch.
    unique_values = ops.unique(seqs.flatten())[0]
    
    # The actual nodes are the unique tokens, EXCLUDING the padding token.
    node_tokens = unique_values[unique_values != padding_id]
    
    # Create a stable lookup table based on the full vocab size. This is more efficient
    # and avoids the previous bug of overwriting incorrect indices.
    lookup = ops.zeros(vocab_size, dtype=ms.int32)
    
    # Map each node token to a sequential index (0, 1, 2, ...).
    lookup[node_tokens] = ops.arange(len(node_tokens), dtype=ms.int32)
    
    # Use the lookup table to convert the input sequences to node indices.
    index = lookup[seqs]
    
    # Crucially, find where the original padding was and set the index to -1 there.
    index[seqs == padding_id] = -1
    
    # The initial leaf indices are simply the range of the created nodes.
    leaf_inds = ops.arange(len(node_tokens), dtype=ms.int32)
    
    return index, node_tokens, leaf_inds

def get_complete(frontiers: Tensor):
    """Checks which sequences in the batch are not yet fully composed."""
    return ~ops.all(frontiers[:, 1:] == -1, axis=1)

def get_sims(nodes: Tensor, index: Tensor):
    """
    Calculates cosine similarities between adjacent nodes in each frontier.
    Note: @torch.no_grad() is removed as it's the default behavior in inference.
    
    # empty to tensor to perform the similarity check
    sims = ops.full((index.shape[0], index.shape[1], nodes.shape[1]), -np.inf, dtype=nodes.dtype)
    # fill sims with the actual node embedding values
    mask = index != -1
    sims[mask] = nodes[index[mask]]
    
    # take similarity between adjacent nodes in each frontier 
    cosines = ops.cosine_similarity(sims[:, :-1, :], sims[:, 1:, :], dim=2)
    
    # mask the padded values
    pad_mask = (sims == -np.inf).all(axis=2)[:, 1:]
    cosines = ops.masked_fill(cosines, pad_mask, -np.inf)
    
    # get the most similar pairs in each frontier
    max_sim = ops.argmax(cosines, axis=1)
    
    # additionally get the retrieval tensor (max_sim, max_sim + 1)
    retrieval = ops.cat((ops.expand_dims(max_sim, 0), ops.expand_dims(max_sim + 1, 0)), axis=0).T.reshape(-1)
    return ops.cast(max_sim, ms.int32), ops.cast(retrieval, ms.int32)
    """
    # 1. Create a "padded" nodes tensor by adding a row of -np.inf.
    #    This row will correspond to the padding index (-1).
    padding_row = ops.full((1, nodes.shape[1]), -np.inf, dtype=nodes.dtype)
    nodes_padded = ops.cat((nodes, padding_row), axis=0)
    
    # 2. Use the index tensor to directly gather from the padded nodes tensor.
    #    MindSpore correctly handles negative indexing (-1 picks the last row),
    #    so this single operation replaces the entire 'create sims, mask, and fill' logic.
    #    The result 'sims' has the shape (batch, seq_len, embedding_dim).
    sims = nodes_padded[index]
    
    # 3. Calculate cosine similarities between adjacent nodes in each frontier.
    cosines = ops.cosine_similarity(sims[:, :-1, :], sims[:, 1:, :], dim=2)
    
    # 4. Mask the padded values to ensure they are not chosen by argmax.
    #    We check for vectors that are all -np.inf, which indicates a padded position.
    pad_mask = (sims == -np.inf).all(axis=2)[:, 1:]
    cosines = ops.masked_fill(cosines, pad_mask, -np.inf)
    
    # 5. Get the most similar pairs in each frontier.
    max_sim = ops.argmax(cosines, 1)
    
    # 6. Additionally get the retrieval tensor (max_sim, max_sim + 1).
    retrieval = ops.cat((ops.expand_dims(max_sim, 0), ops.expand_dims(max_sim + 1, 0)), axis=0).T.reshape(-1)
    return ops.cast(max_sim, ms.int32), ops.cast(retrieval, ms.int32)

def reduce_frontier(index: Tensor, completion_mask: Tensor, range_tensor: Tensor, max_indices: Tensor):
    """
    Reduces the frontier by removing the left child of a composed pair.
    Note: @torch.no_grad() is removed.
    """
    """
    # create a mask to perform frontier reduction
    batch_remaining_mask = ops.ones_like(index, dtype=ms.bool_)
    
    # remove left child of composed sequences
    if completion_mask.any():
        batch_remaining_mask[range_tensor[completion_mask], max_indices[completion_mask]] = False
        
    # for completed sequences remove padding element so shapes fit
    '''
    non_completed_mask = ops.where(completion_mask == False)[0]
    if non_completed_mask.numel() != 0:
        batch_remaining_mask[non_completed_mask, -1] = False
    '''
    # [FIX] MindSpore's ops.where is ternary. Use ops.nonzero to find indices from a condition.
    # The condition is `completion_mask == False`. We find the indices of `True` values in this result
    # and flatten to get a 1D tensor of indices.
    non_completed_mask = ops.nonzero(completion_mask == False).flatten()
    if non_completed_mask.size != 0:
        batch_remaining_mask[non_completed_mask, -1] = False    
    # reduce the index tensor
    # MindSpore doesn't directly support masked select on 2D to reshape.
    # We flatten, select, and then reshape.
    new_size = index.shape[0] * (index.shape[1] - 1)
    index = index[batch_remaining_mask].reshape(index.shape[0], -1)
    return index
    """
    """
    [REFACTORED & OPTIMIZED] Reduces the frontier by removing one node from each sequence
    using pure tensor operations to create a static graph, eliminating performance warnings.
    """
    '''
    """
    [REFACTORED & OPTIMIZED] Reduces the frontier by replacing one node from each sequence
    with a padding value (-1) and then sorting to maintain a fixed tensor shape. This
    creates a static graph and is highly performant on Ascend.
    """
    # For each sequence, determine the column index of the node to remove.
    # If completed, remove the left child of the composed pair (`max_indices`).
    # If not completed, remove the right-most valid node.
    last_valid_node = (index != -1).sum(axis=1) - 1
    cols_to_replace = ops.where(completion_mask, max_indices, last_valid_node)

    # Create the full 2D indices for the scatter operation.
    indices_to_replace = ops.stack((range_tensor, cols_to_replace), axis=1)

    # The value to scatter is -1 (padding) for all positions to be "removed".
    updates = ops.full((index.shape[0],), -1, dtype=index.dtype)

    # Use tensor_scatter_update to place -1 at the target locations.
    index_with_holes = ops.tensor_scatter_update(index, indices_to_replace, updates)

    # Sort each row in descending order. This is a highly efficient way to "compact"
    # the tensor, moving all valid positive indices to the left and all -1 padding
    # values to the right, without changing the tensor's shape.
    new_index, _ = ops.sort(index_with_holes, axis=1, descending=True)
    
    return new_index
    '''
    # For each sequence in the batch, determine the column index of the node to remove.
    # If the sequence composition was completed (`completion_mask` is True), remove the left child of the composed pair (`max_indices`).
    # If the sequence was not completed (`completion_mask` is False), remove the last node in its frontier to keep dimensions consistent.
    cols_to_mask = ops.where(
        completion_mask,
        max_indices,
        ops.full(completion_mask.shape, index.shape[1] - 1, dtype=ms.int32)
    )

    # Create the full 2D indices for the scatter operation.
    # The row indices are simply 0, 1, ..., batch_size-1.
    indices_to_mask = ops.stack((range_tensor, cols_to_mask), axis=1)

    # The values to scatter are `False` for all positions to be masked.
    updates = ops.zeros(index.shape[0], dtype=ms.bool_)

    # Start with a mask of all `True` and apply the updates in a single, efficient operation.
    # `tensor_scatter_update` is the canonical way to perform this scatter operation in a graph-friendly way.
    batch_remaining_mask = ops.ones_like(index, dtype=ms.bool_)
    batch_remaining_mask = ops.tensor_scatter_update(batch_remaining_mask, indices_to_mask, updates)

    # Apply the final mask to the index tensor and reshape to the new, smaller sequence length.
    index = index[batch_remaining_mask].reshape(index.shape[0], -1)
    return index