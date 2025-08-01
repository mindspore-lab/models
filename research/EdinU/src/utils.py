# utils_ms.py
# dataloading, seed setting and general helper functions for MindSpore
import mindspore as ms
import mindspore.dataset as ds
import pandas as pd
from bpemb import BPEmb
from tqdm import tqdm
import numpy as np
import random

def sent_to_bpe(sent, bpe):
    """Converts a sentence to a NumPy array of BPE IDs."""
    encoded = bpe.encode_ids(sent)
    return np.array(encoded, dtype=np.int32)


def process_dataset(data_path, lang):
    """Processes a raw text file into a list of NumPy arrays."""
    dataset = []
    bpemb_en = BPEmb(lang=lang, vs=25000, dim=100)
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[:-1]):  # skip the last line which is empty
            line = line.strip('\n')
            emb = sent_to_bpe(line, bpemb_en)
            if 3 < emb.shape[0] <= 200:
                dataset.append(emb)
    return dataset

class StrAEDataset:
    """A source dataset class compatible with MindSpore's GeneratorDataset."""
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return self.n_samples


def collate_fn(*args):
    """
    [CORRECTED] Pads sequences to the max length in a batch using a robust *args signature.
    """
    # Use *args to robustly capture all arguments from the dataset pipeline.
    # We only need the first argument, which is the list of sequences for our 'data' column.
    data = args[0]
    max_len = max([s.shape[0] for s in data])
    padding_value = 25000
    
    padded_seqs = []
    for seq in data:
        pad_len = max_len - seq.shape[0]
        padding = np.full((pad_len,), padding_value, dtype=seq.dtype)
        padded_seqs.append(np.concatenate([seq, padding]))
        
    # per_batch_map expects a tuple of numpy arrays, one for each output column.
    return (np.stack(padded_seqs),)


def process_sts_dataset(data_path, lang):
    """Processes a CSV file for STS tasks into lists of NumPy arrays and scores."""
    df = pd.read_csv(data_path)
    bpemb_en = BPEmb(lang=lang, vs=25000, dim=100)
    sents1 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sent1']]
    sents2 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sent2']]
    scores = [np.array(x, dtype=np.float32) for x in df['score']]
    dataset = [(sents1[x], sents2[x], scores[x]) for x in range(len(sents1))]
    return dataset


class STSDataset:
    """A source dataset class for STS tasks, compatible with GeneratorDataset."""
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index][0], self.sequences[index][1], self.sequences[index][2]

    def __len__(self):
        return self.n_samples


def collate_fn_sts(*args):
    """
    [CORRECTED] Pads sequence pairs and stacks scores using a robust *args signature.
    """
    # Use *args to robustly capture all arguments from the dataset pipeline.
    # We need the first three arguments for our columns. Any extra args are ignored.
    sents_1, sents_2, scores = args[0], args[1], args[2]
    padding_value = 25000
    
    max_len_1 = max([s.shape[0] for s in sents_1])
    padded_sents_1 = []
    for seq in sents_1:
        pad_len = max_len_1 - seq.shape[0]
        padding = np.full((pad_len,), padding_value, dtype=seq.dtype)
        padded_sents_1.append(np.concatenate([seq, padding]))

    max_len_2 = max([s.shape[0] for s in sents_2])
    padded_sents_2 = []
    for seq in sents_2:
        pad_len = max_len_2 - seq.shape[0]
        padding = np.full((pad_len,), padding_value, dtype=seq.dtype)
        padded_sents_2.append(np.concatenate([seq, padding]))

    return np.stack(padded_sents_1), np.stack(padded_sents_2), np.stack(scores)

        
def create_sts_dataloader(data_path, batch_size, shuffle=False, lang='en'):
    """[CORRECTED] Creates a MindSpore DataLoader for STS tasks using the idiomatic padded_batch API."""
    data = process_sts_dataset(data_path, lang)
    dataset = STSDataset(data)
    generator_dataset = ds.GeneratorDataset(dataset, column_names=["sents_1", "sents_2", "scores"], shuffle=shuffle)
    
    dataloader = generator_dataset.batch(batch_size, per_batch_map=collate_fn_sts, output_columns=["sents_1", "sents_2", "scores"], num_parallel_workers=4)
    return dataloader


def create_dataloader(data_path, batch_size, shuffle=False, lang='en'):
    """Creates a MindSpore DataLoader for standard language modeling."""
    data = process_sts_dataset(data_path, lang)
    dataset = STSDataset(data)
    generator_dataset = ds.GeneratorDataset(dataset, column_names=["sents_1", "sents_2", "scores"], shuffle=shuffle)
    dataloader = generator_dataset.batch(batch_size, per_batch_map=collate_fn_sts, output_columns=["sents_1", "sents_2", "scores"], num_parallel_workers=4)
    return dataloader


def set_seed(seed=None):
    """Sets the random seed for MindSpore and NumPy."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    ms.set_seed(seed)
    np.random.seed(seed)
    return seed