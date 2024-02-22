# -*- encoding: utf-8 -*-
# here put the import lib
from generators.generator import Generator
from generators.data import BertTrainDataset, BertRecTrainDataset
from mindspore.dataset import GeneratorDataset, RandomSampler, BatchDataset
from utils.utils import unzip_data



class BertGenerator(Generator):

    def __init__(self, args, logger):

        super().__init__(args, logger)
    

    def make_trainloader(self):
        
        train_dataset = unzip_data(self.train, aug=self.args.aug)
        if (self.args.model_name == 'bert4rec_pretrain') | (self.args.model_name == 's3rec_pretrain'):
            train_dataset = BertTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)
        else:
            train_dataset = BertRecTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)

        num_samples = num_samples=train_dataset.__len__()
        train_dataloader = GeneratorDataset(train_dataset,
                                            sampler=RandomSampler(num_samples=num_samples),
                                            column_names=["tokens", "labels", "neg_labels", "pos"],
                                            )
        train_dataloader = train_dataloader.batch(batch_size=self.bs, num_parallel_workers=self.num_workers)
    
        return train_dataloader









