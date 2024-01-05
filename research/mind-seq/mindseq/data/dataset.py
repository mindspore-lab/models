import logging
import os

import numpy as np
import pandas as pd

# from setting import data_path
from ..utils.helper import Scaler, asym_adj
from ..utils.timefeatures import time_features_allot as time_features
from mindspore.dataset import GeneratorDataset
import warnings
warnings.filterwarnings('ignore')

# data_path = '../dataset'
# param_path = '../param'

class Dataset_ST:
    def __init__(self, data_path, path, train_prop, test_prop,
                 tag='train', seq_len=12, pred_len=12):
        self.data_path = data_path
        self._path = path
        self._tag = tag
        
        self._train_prop = train_prop
        self._test_prop = test_prop
        
        self._seq_len = seq_len
        self._pred_len = pred_len
        
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self._path, 'traffic.csv'))
        
        # timefeatures
        timestamps = time_features(df_raw[['date']], freq='t')
        
        # remove date
        df_raw = df_raw[df_raw.columns[1:]]
        
        # fill nan
        df_rec = df_raw.copy()
        df_rec = df_rec.replace(0, np.nan)
        df_rec = df_rec.bfill() #.fillna(method='pad')
        df_rec = df_rec.ffill() #.fillna(method='bfill')
        
        # data split
        num_samples = len(df_rec)
        num_train = round(num_samples * self._train_prop)
        num_test = round(num_samples * self._test_prop)
        num_val = num_samples-num_train-num_test
        
        # set scaler
        train_data = df_rec.values[:num_train]
        self.scaler = Scaler(train_data, missing_value=0.)
        borders = {
            'train':[0, num_train],
            'valid':[num_train,num_train+num_val],
            'test':[num_samples-num_test,num_samples]
        }
        border1, border2 = borders[self._tag][0], borders[self._tag][1]
        data = df_rec.values[border1:border2]
        data = self.scaler.transform(data)
        
        self.data_x = data
        self.data_y = df_rec.values[border1:border2] # df_raw.values[border1:border2]
        self.data_t = timestamps[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self._seq_len
        r_begin = s_end; r_end = r_begin + self._pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_t[s_begin:s_end]
        seq_y_mark = self.data_t[r_begin:r_end]
        
        nodes = seq_x.shape[-1]
        seq_x = np.expand_dims(seq_x, -1) # [seq_len, nodes, 1]
        seq_y = np.expand_dims(seq_y, -1) # [pred_len, nodes, 1]
        seq_x_mark = np.tile(np.expand_dims(seq_x_mark, -2), [1,nodes,1]) # [seq_len, nodes, timefeatures_dim]
        seq_y_mark = np.tile(np.expand_dims(seq_y_mark, -2), [1,nodes,1]) # [pred_len, nodes, timefeatures_dim]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self._seq_len - self._pred_len + 1


class TrafficDataset:
    def __init__(self, data_path, path, train_prop, test_prop,
                 num_sensors, normalized_k=0.1, adj_type='distance',
                 in_length=12, out_length=12, batch_size=32,device_num=None,rank_id=None):
        logging.info('initialize %s DataWrapper', path)
        self.data_path = data_path
        self._path = path
        self._train_prop = train_prop
        self._test_prop = test_prop
        self._num_sensors = num_sensors
        self._normalized_k = normalized_k
        self._adj_type = adj_type
        self._in_length = in_length
        self._out_length = out_length
        self._batch_size = batch_size
        self._device_num = device_num
        self._rank_id = rank_id

        self.build_graph()

    def build_graph(self):
        logging.info('initialize graph')

        self.adj_mats = self.read_adj_mat()
        
        for dim in range(self.adj_mats.shape[-1]):
            # normalize adj_matrix
            if dim%2 != 0:
                self.adj_mats[:, :, dim][self.adj_mats[:,:,dim]==np.inf] = 0.
            else:
                values = self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] != np.inf].flatten()       
                self.adj_mats[:, :, dim] = np.exp(-np.square(self.adj_mats[:, :, dim] / (values.std() + 1e-8)))
                self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] < self._normalized_k] = 0
            # transfer adj_matrix
            self.adj_mats[:, :, dim] = asym_adj(self.adj_mats[:, :, dim])    
        
        if self._adj_type=='distance':
            self.adj_mats = self.adj_mats[:,:,::2]
        elif self._adj_type=='connect':
            self.adj_mats = self.adj_mats[:,:,1::2]
        else:
            pass

        dataset = Dataset_ST(
            data_path = self.data_path,
            path = self._path,
            train_prop = self._train_prop,
            test_prop = self._test_prop,
            tag = 'train',
            seq_len = self._in_length,
            pred_len = self._out_length
        )
        self.scaler = dataset.scaler
        
    def get_dataloader(self, tag='train', batch_size=None, num_workers=None):
        logging.info('load %s inputs & labels [start]', tag)
        
        dataset = Dataset_ST(
            data_path = self.data_path,
            path = self._path,
            train_prop = self._train_prop,
            test_prop = self._test_prop,
            tag = tag,
            seq_len = self._in_length,
            pred_len = self._out_length
        )
        if tag != 'test':
            data_set = GeneratorDataset(source=dataset, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],
                                            num_shards=self._device_num, shard_id=self._rank_id)
        else:
            data_set = GeneratorDataset(source=dataset, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"])
        data_shuffle = True if tag != 'test' else False
        data_batch_size = batch_size or self._batch_size
        if data_shuffle:
            data_set = data_set.shuffle(data_set.get_dataset_size())
        data_set = data_set.batch(batch_size = data_batch_size, drop_remainder=False)
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: ({},{},{},{})'.format(len(dataset), self._in_length, self._num_sensors, 1))
        logging.info('label shape: ({},{},{},{})'.format(len(dataset), self._out_length, self._num_sensors, 1))
        return data_set
        
    def read_adj_mat(self):
        cache_file = os.path.join(self.data_path, self._path, 'sensor_graph/adjacent_matrix_cached.npz')
        try:
            arrays = np.load(cache_file)
            g = arrays['g']
            logging.info('load adj_mat from the cached file [ok]')
        except:
            logging.info('load adj_mat from the cached file [fail]')
            logging.info('load adj_mat from scratch')
            
            # read idx
            with open(os.path.join(self.data_path, self._path, 'sensor_graph/graph_sensor_ids.txt')) as f:
                ids = f.read().strip().split(',')
            idx = {}
            for i, id in enumerate(ids):
                idx[id] = i
            
            # read graph
            graph_csv = pd.read_csv(os.path.join(self.data_path, self._path, 'sensor_graph/distances.csv'),
                                    dtype={'from': 'str', 'to': 'str'})
            g = np.zeros((self._num_sensors, self._num_sensors, 2))
            g[:] = np.inf

            for k in range(self._num_sensors): g[k, k] = 0
            for row in graph_csv.values:
                if row[0] in idx and row[1] in idx:
                    g[idx[row[0]], idx[row[1]], 0] = row[2]  # distance
                    g[idx[row[0]], idx[row[1]], 1] = 1  # hop

            g = np.concatenate([g, np.transpose(g, (1, 0, 2))], axis=-1)
            np.savez_compressed(cache_file, g=g)
            logging.info('save graph to the cached file [ok]')
        return g
