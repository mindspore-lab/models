import json
from mindspore.dataset import GeneratorDataset
from data_utils.vocab import *
# from mindnlp.transformers import AutoTokenizer
from os.path import join

def load_file(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class LAMADataset:
    def __init__(self, dataset_type, data, tokenizer, args):
        self.args = args
        self.data = []
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.x_hs, self.x_ts = [], []
        vocab = get_vocab_by_strategy(args, tokenizer)

        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            self.x_ts.append(d['obj_label'])
            self.x_hs.append(d['sub_label'])
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['sub_label'], self.data[index]['obj_label']

    def create_dataset(self, batch_size=1, shuffle=False):
        """
        Create a MindSpore GeneratorDataset from this dataset
        """
        def generator():
            for i in range(len(self)):
                yield self.__getitem__(i)
        
        column_names = ["sub_label", "obj_label"]
        dataset = GeneratorDataset(
            source=generator,
            column_names=column_names,
            shuffle=shuffle,
            num_parallel_workers=1
        )
        
        if batch_size > 1:
            dataset = dataset.batch(batch_size)
        
        return dataset

class LAMADataset_new:
    def __init__(self, dataset_type, data, tokenizer, args):
        self.args = args
        self.data = []
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.x_hs, self.x_ts = [], []
        
        # Load shared vocabulary
        vocab_path = join(args.data_dir, '29k-vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            shared_vocab = json.load(f)
        
        vocab = shared_vocab['bert-base-cased']
        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            self.x_ts.append(d['obj_label'])
            self.x_hs.append(d['sub_label'])
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['sub_label'], self.data[index]['obj_label']

    def create_mindspore_dataset(self, batch_size=1, shuffle=False):
        """
        Convert to MindSpore GeneratorDataset
        """
        def data_generator():
            for i in range(len(self)):
                yield self.__getitem__(i)
        
        dataset = GeneratorDataset(
            source=data_generator,
            column_names=["sub_label", "obj_label"],
            shuffle=shuffle,
            num_parallel_workers=4
        )
        
        if batch_size > 1:
            dataset = dataset.batch(batch_size)
        
        return dataset

