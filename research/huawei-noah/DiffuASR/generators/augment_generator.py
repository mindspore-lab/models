# -*- encoding: utf-8 -*-
# here put the import lib
import numpy as np
from generators.generator import Generator
from utils.utils import unzip_data, concat_data, concat_aug_data
from mindspore.dataset import Dataset, GeneratorDataset, SequentialSampler


class AugmentGenerator(Generator):

    def __init__(self, args, logger, device):
        
        super().__init__(args, logger, device)
        self.aug_file = args.aug_file


    def make_augmentloader(self):

        #TODO: two data for valid and test respectively?
        #aug_dataset = unzip_data(self.train, aug=False, aug_num=0)
        aug_dataset = concat_aug_data([self.train, self.valid])
        #aug_dataset = concat_data([self.train, self.valid])
        aug_dataset = AugmentDataset(aug_dataset, self.item_num, self.args.max_len)

        aug_dataloader = GeneratorDataset(aug_dataset,
                                    sampler=SequentialSampler(aug_dataset),
                                    batch_size=500,
                                    num_workers=self.num_workers)

        return aug_dataloader

    
    def save_aug(self, aug_data):

        aug_data = aug_data.tolist()

        res_data = []

        for i in range(len(aug_data)):

            per_data = aug_data[i] + self.train[i+1] + self.valid[i+1] + self.test[i+1]
            res_data.append(per_data)
        
        with open('./data/%s/aug/%s.txt' % (self.dataset, self.aug_file), 'w') as f:
        
            for user in range(len(aug_data)):

                for item in res_data[user]:

                    f.write('%s %s\n' % (int(user+1), int(item)))



class AugmentDataset(Dataset):

    def __init__(self, data, item_num, max_len) -> None:

        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len

    
    def __len__(self):

        return len(self.data)

    
    def __getitem__(self, index):

        inter = self.data[index]
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in inter: # ti get the reversed sequence
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - len(inter)
            positions = list(range(1, len(inter)+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions


