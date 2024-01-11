import numpy as np
import pandas as pd
import torch.utils.data

class DoubanMusic_split():

    def __init__(self, dataset_path='/Douban/Data/train/douban_music_train.csv', y=0):
        
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:4]
        data=data[data[:,2]==y] #select y=0 and y=1
        
        self.items = data[:, :2].astype(int) #- 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),0)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = data[:, 2].astype(int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 0

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index]
        
class DoubanMusic_sparse_split():

    def __init__(self, dataset_path='/Douban/Data/train/douban_music_sparse_train.csv', y=0):
        
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:4]
        
        data=data[data[:,2]==y] #select y=0 and y=1
        
        self.items = data[:, :2].astype(int) #- 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),0)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = data[:, 2].astype(int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.items[:,1] = self.items[:,1] + 0

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index]

class DoubanBook_split():

    def __init__(self, dataset_path='/Douban/Data/douban_book/ratings.dat', y=0):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:4]
        data=data[data[:,2]==y] #select y=0 and y=1
        
        self.items = data[:, :2].astype(int) #- 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),1)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = data[:, 2].astype(int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        #self.items[:,1] = self.items[:,1] + 5567
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index]


class DoubanMovie_split():

    def __init__(self, dataset_path='/Douban/Data/douban_movie/ratings.dat', y=0):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:4]
        data=data[data[:,2]==y] #select y=0 and y=1
        
        self.items = data[:, :2].astype(int) #- 1  # -1 because ID begins from 1
        domain_id = np.full((self.items.shape[0],1),2)
        self.items = np.concatenate([domain_id,self.items],1)
        
        self.targets = data[:, 2].astype(int)
        self.field_dims = np.ndarray(shape=(3,), dtype=int)
        #self.field_dims[0] = np.max(self.items[:,0])+1
        #self.field_dims[1] = np.max(self.items[:,1])+1
        self.field_dims[0] = 3
        self.field_dims[1] = 2718
        self.field_dims[2] = 5567 + 6777 + 9565
        #print(self.field_dims)
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        #self.items[:,1] = self.items[:,1] + 5567 + 6777

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index]
