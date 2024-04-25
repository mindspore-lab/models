import os
import random
import math
import datasets
from mindspore.dataset import GeneratorDataset
from arguments import DataArguments
from mindspore.communication import get_rank, get_group_size
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
random.seed(1)


class TrainDatasetForEmbedding():
    def __init__(
            self,
            args: DataArguments,
            tokenizer: None
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        item = int(item)
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]

        query_tokenized = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.args.query_max_len, return_tensors='ms')
        passages_tokenized = self.tokenizer(passages, padding='max_length', truncation=True, max_length=self.args.passage_max_len, return_tensors='ms')

        return (query_tokenized["input_ids"],
                query_tokenized["token_type_ids"],
                query_tokenized["attention_mask"],
                passages_tokenized["input_ids"],
                passages_tokenized["token_type_ids"],
                passages_tokenized["attention_mask"])


def process_data(source, batch_size=1, shuffle=True, use_parallel=False):
    column_names = ["query_input_ids", "query_token_type_ids", "query_attention_mask",
                  "passages_input_ids", "passages_token_type_ids", "passages_attention_mask"]
    if use_parallel:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle,
                                   num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

