import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import mindspore as ms
from mindspore import Tensor, ops
import variables as var
from scipy.io import loadmat
import faiss
from sklearn.neighbors import NearestNeighbors
class Data:
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        info = []
        for key, value in self.__dict__.items():
            info.append(f'{key}={str(value.shape)}')
        info = ', '.join(info)
        return f'Data({info})'

    def num_nodes(self):
        return self.x.shape[0] if self.x is not None else None

    def is_directed(self):
        return not (self.edge_index[0] == self.edge_index[1].flip([0])).all()

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, key, value.asnumpy())
        return self
########################################### NEGATIVE SAMPLE FUNCTIONS################################################
def negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, sample_type, proportion, epsilon):
    
    # training set negative samples
    neg_train_x, neg_train_y = generate_negative_samples(train_x, sample_type, proportion, epsilon)
    # validation set negative samples
    neg_val_x, neg_val_y = generate_negative_samples(val_x, sample_type, proportion, epsilon)
    
    # concat data
    x = np.vstack((train_x,neg_train_x,val_x,neg_val_x,test_x))
    y = np.hstack((train_y,neg_train_y,val_y,neg_val_y,test_y))
    x = x.astype('float32')
    y = y.astype('float32')
    # all training set
    train_mask = np.hstack((np.ones(len(train_x)),np.ones(len(neg_train_x)),
                            np.zeros(len(val_x)),np.zeros(len(neg_val_x)),
                            np.zeros(len(test_x))))
    # all validation set
    val_mask = np.hstack((np.zeros(len(train_x)),np.zeros(len(neg_train_x)),
                          np.ones(len(val_x)),np.ones(len(neg_val_x)),
                          np.zeros(len(test_x))))
    # all test set
    test_mask = np.hstack((np.zeros(len(train_x)),np.zeros(len(neg_train_x)),
                           np.zeros(len(val_x)),np.zeros(len(neg_val_x)),
                           np.ones(len(test_x))))
    # normal training points
    neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(neg_train_x)), 
                               np.zeros(len(val_y)), np.zeros(len(neg_val_x)),
                               np.zeros(len(test_y))))
    train_mask = train_mask.astype('float32')
    val_mask = val_mask.astype('float32')
    test_mask = test_mask.astype('float32')
    neighbor_mask = neighbor_mask.astype('float32')
    
    # find k nearest neighbours (idx) and their distances (dist) to each points in x within neighbour_mask==1
    dist, idx = find_neighbors(x, y, neighbor_mask, k)

    x = Tensor(x, ms.float32)
    y = Tensor(y, ms.float32)
    neighbor_mask = Tensor(neighbor_mask, ms.float32)
    train_mask = Tensor(train_mask, ms.float32)
    val_mask = Tensor(val_mask, ms.float32)
    test_mask = Tensor(test_mask, ms.float32)
    return x,y,neighbor_mask,train_mask,val_mask,test_mask, dist, idx
    # return x.astype('float32'), y.astype('float32'), neighbor_mask.astype('float32'), train_mask.astype('float32'), val_mask.astype('float32'), test_mask.astype('float32'), dist, idx

def generate_negative_samples(x, sample_type, proportion, epsilon):
    n_samples = int(proportion * len(x))
    n_dim = x.shape[-1]
    
    # Convert x to MindSpore tensor
    x_ms = Tensor(x, ms.float32)
    
    # Generate random matrix (randmat) with MindSpore
    randmat = ops.less(Tensor(np.random.rand(n_samples, n_dim), ms.float32), 0.3).astype(ms.float32)
    
    # Generate uniform samples (rand_unif) with MindSpore
    rand_unif = epsilon * (1 - 2 * ops.StandardNormal()((n_samples, n_dim)))
    
    # Generate subspace perturbation samples (rand_sub) with MindSpore
    tile_x_ms = ops.Tile()(x_ms, (proportion, 1))
    rand_noise = epsilon * ops.StandardNormal()((n_samples, n_dim))
    rand_sub = tile_x_ms + randmat * rand_noise
    
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    elif sample_type == 'SUBSPACE':
        neg_x = rand_sub
    elif sample_type == 'MIXED':
        # Concatenate uniform and subspace samples
        neg_x = ops.Concat(axis=0)((rand_unif, rand_sub))
        indices = np.random.choice(np.arange(len(neg_x)), size=n_samples, replace=False)
        neg_x = ops.Gather()(neg_x, Tensor(indices, ms.int32), 0)
    
    neg_y = ops.OnesLike()(neg_x[:, 0])
    
    # Convert back to numpy
    neg_x_np = neg_x.asnumpy()
    neg_y_np = neg_y.asnumpy()
    
    return neg_x_np.astype('float32'), neg_y_np.astype('float32')
################################### GRAPH FUNCTIONS ###############################################     
# find the k nearest neighbours of all x points out of the neighbour candidates
def find_neighbors(x, y, neighbor_mask, k):
    
    # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])
    # add nearest neighbour candidates
    # index.add(x[neighbor_mask==1])
    x_neighbors_1 = x[neighbor_mask == 1].astype('float32') 
    x_neighbors_0 = x[neighbor_mask == 0].astype('float32') 
    index.add(x_neighbors_1)

    # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist_train, idx_train = index.search(x_neighbors_1, k = k+1)
    # remove 1st nearest neighbours to remove self loops
    dist_train, idx_train = dist_train[:,1:], idx_train[:,1:]
    # distances and idx of neighbour points for the non-neighbour candidates
    dist_test, idx_test = index.search(x_neighbors_0, k = k)
    #concat
    dist = np.vstack((dist_train, dist_test))
    idx = np.vstack((idx_train, idx_test))
    
    return dist, idx

# create graph object out of x, y, distances and indices of neighbours
def build_graph(x, y, dist, idx):
    
    # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)
    idx_source = np.repeat(np.arange(len(x)),dist.shape[-1]).astype('int32')
    idx_source = np.expand_dims(idx_source,axis=0)

    # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
    idx_target = idx.flatten()
    idx_target = np.expand_dims(idx_target,axis=0).astype('int32')
    
    #stack source and target indices
    idx = np.vstack((idx_source, idx_target))

    # edge weights
    attr = dist.flatten()
    attr = np.sqrt(attr)
    attr = np.expand_dims(attr, axis=1)
    
    # into tensors
    # x = torch.tensor(x, dtype = torch.float32)
    # y = torch.tensor(y,dtype = torch.float32)
    # idx = torch.tensor(idx, dtype = torch.long)
    # attr = torch.tensor(attr, dtype = torch.float32)

    x = Tensor(x, ms.float32)
    y = Tensor(y, ms.float32)
    idx = Tensor(idx, ms.int32)
    attr = Tensor(attr, ms.float32)
    #build  geometric Data object
    data = Data(x = x, edge_index = idx, edge_attr = attr, y = y)
    
    return data

########################################## DATASET FUNCTIONS ####################################   
#  
# split training data into train set and validation set
def split_data(seed, all_train_x, all_train_y, all_test_x, all_test_y):
    np.random.seed(seed)

    val_idx = np.random.choice(np.arange(len(all_train_x)),size = int(0.15*len(all_train_x)), replace = False)
    val_mask = np.zeros(len(all_train_x))
    val_mask[val_idx] = 1
    val_x = all_train_x[val_mask == 1]; val_y = all_train_y[val_mask == 1]
    train_x = all_train_x[val_mask == 0]; train_y = all_train_y[val_mask == 0]
    
    scaler = MinMaxScaler()
    scaler.fit(train_x[train_y == 0])
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
   
    if all_test_x is None:
        test_x = val_x
        test_y = val_y
    
    test_x = scaler.transform(all_test_x)
    test_y = all_test_y
	
    train_x = Tensor(train_x, ms.float32)
    train_y = Tensor(train_y, ms.float32)
    val_x = Tensor(val_x, ms.float32)
    val_y = Tensor(val_y, ms.float32)
    test_x = Tensor(test_x, ms.float32)
    test_y = Tensor(test_y, ms.float32)
    return train_x,train_y,val_x,val_y,test_x,test_y
    # return train_x.astype('float32'), train_y.astype('float32'), val_x.astype('float32'), val_y.astype('float32'),  test_x.astype('float32'), test_y.astype('float32')


#load data
def load_dataset(dataset,seed):     
    np.random.seed(seed)    
    
    if dataset == 'MI-V':
        df = pd.read_csv("data/MI/experiment_01.csv")
        for i in ['02','03','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
        normal_idx = np.ones(len(df))
        for i in ['06','08','09','10']:
            data = pd.read_csv("data/MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)        
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))
        
    elif dataset == 'MI-F':
        df = pd.read_csv("data/mi/experiment_01.csv")
        for i in ['02','03','06','08','09','10','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
        normal_idx = np.ones(len(df))
        for i in ['04', '05', '07', '16']: 
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))  
        
    elif dataset in ['OPTDIGITS', 'PENDIGITS','SHUTTLE']:   
        if dataset == 'SHUTTLE':
            data = loadmat("data/SHUTTLE/shuttle.mat")
        elif dataset == 'OPTDIGITS':
            data = loadmat("data/optdigits/optdigits.mat")
        elif dataset == 'PENDIGITS':
            data = loadmat('data/PENDIGITS/pendigits.mat')  
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data= data[label == 0]
        normal_label = label[label==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))
        test_y = np.concatenate((normal_label[test_idx],anom_label))
        
    elif dataset in ['THYROID','HRSS']:
        if dataset == 'THYROID':
            data = pd.read_csv('data/THYROID/annthyroid_21feat_normalised.csv').to_numpy()
        if dataset == 'HRSS':
            data = pd.read_csv('/media/data2/xidian/ww/LUNAR-main/data/data/HRSS/HRSS.csv').to_numpy()
        label = data[:,-1].astype('float32').squeeze()
        data = data[:,:-1].astype('float32')
        normal_data= data[label == 0]
        normal_label = label[label==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))
        test_y = np.concatenate((normal_label[test_idx],anom_label)) 
        
    elif dataset == 'SATELLITE':
        data = loadmat('data/SATELLITE/satellite.mat')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label ==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        train_idx = np.random.choice(np.arange(0,len(normal_data)), 4000, replace = False)
        test_idx = np.setdiff1d(np.arange(0,len(normal_data)), train_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = normal_data[test_idx]
        test_y = normal_label[test_idx]
        test_idx = np.random.choice(np.arange(0,len(anom_data)), int(len(test_x)), replace = False)
        test_x = np.concatenate((test_x,anom_data[test_idx]))
        test_y = np.concatenate((test_y, anom_label[test_idx])) 
                
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(seed, all_train_x = train_x, all_train_y = train_y, all_test_x = test_x, all_test_y = test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y       
