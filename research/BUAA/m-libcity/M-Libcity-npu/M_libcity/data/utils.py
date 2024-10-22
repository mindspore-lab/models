import importlib
import os
import mindspore.dataset as ds
import numpy as np
import pandas as pd
from mindspore.communication.management import get_rank, get_group_size #多卡


def l2l(origin_len):
    """
        输入的origin_len包括4个key，'history_loc', 'history_tim', 'current_loc', 'current_tim'
        该方法需要将其转换成history_len,current_len
    """
    loc_len = origin_len['current_loc']
    history_len = origin_len['history_loc']
    return loc_len,history_len

def padding(pad_item, pad_max_len, feature_name, data):
    """
    只提供对一维数组的特征进行补齐
    """
    pad_len = {}
    f2indx={}
    origin_len={}

    pad_max_len = pad_max_len if pad_max_len is not None else {}
    if pad_max_len=={} and "history_loc" in pad_item:
        for key in pad_item:
            pad_max_len[key]=100

    pad_item = pad_item if pad_item is not None else {}
    for indx,key in enumerate(feature_name):
        f2indx[key]=indx
        if key in pad_item:
            pad_len[key] = 0
            origin_len[key]=[]

    for i in range(len(data)):
        for indx,key in enumerate(pad_item):
            findx=f2indx[key]
            # 需保证 item 每个特征的顺序与初始化时传入的 feature_name 中特征的顺序一致
            origin_len[key].append(len(data[i][findx]))
            if pad_len[key] < len(data[i][findx]):
                # 保持 pad_len 是最大的
                pad_len[key] = len(data[i][findx])

    for indx,key in enumerate(pad_item):
        # 只对在 pad_item 中的特征进行补齐
        findx=f2indx[key]
        max_len = pad_len[key]
        if key in pad_max_len:
            max_len = min(pad_max_len[key], max_len)
        for i in range(len(data)):
            if len(data[i][findx]) < max_len:
                data[i][findx] += [pad_item[key]] * \
                                (max_len - len(data[i][findx]))
            else:
                # 截取的原则是，抛弃前面的点
                # 因为是时间序列嘛
                data[i][findx] = data[i][findx][-max_len:]
                origin_len[key][i] = max_len
    # 适用于不同版本的代码备份
    # for indx,key in enumerate(pad_item):
    #     findx=f2indx[key]
    #     max_len = pad_len[key]
    #     if key in pad_max_len:
    #         max_len = min(pad_max_len[key], max_len)
        
    #     for i in range(len(data)):
    #         print(len(data[i][findx]))
    #         assert len(data[i][findx]) >= origin_len[key][i]

    if "current_loc" in pad_item:
        loc_len, his_len=l2l(origin_len)
        for i in range(len(data)):
            data[i].append(loc_len[i])
            data[i].append(his_len[i])

    return data


class ListDataset:
    def __init__(self, data,feature_name):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    



def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False,rank_size=1):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """

    if os.getenv('RANK_SIZE'):
        rank_size=int(os.getenv('RANK_SIZE'))
        try:
            assert len(train_data)>(rank_size*batch_size)
            assert len(eval_data)>(rank_size*batch_size)
            assert len(test_data)>(rank_size*batch_size)
        except Exception:
            batch_size=16
            print('Step size is lower than rank size, change the Batch Size to 16.')
    
    train_data_me = []
    for i1 in range(len(train_data)):
        tuple_me = ()
        for i2 in range(len(train_data[i1])):
            ndarry_me = train_data[i1][i2].astype(np.float32) 
            tuple_me = tuple_me + (ndarry_me,)
        train_data_me.append(tuple_me)
    eval_data_me = []
    for i1 in range(len(eval_data)):
        tuple_me = ()
        for i2 in range(len(eval_data[i1])):
            ndarry_me = eval_data[i1][i2].astype(np.float32) 
            tuple_me = tuple_me + (ndarry_me,)
        eval_data_me.append(tuple_me)
    test_data_me = []
    for i1 in range(len(test_data)):
        tuple_me = ()
        for i2 in range(len(test_data[i1])):
            ndarry_me = test_data[i1][i2].astype(np.float32) 
            tuple_me = tuple_me + (ndarry_me,)
        test_data_me.append(tuple_me)
    
    train_dataset = ListDataset(train_data_me,feature_name)
    eval_dataset = ListDataset(eval_data_me,feature_name)
    test_dataset = ListDataset(test_data_me,feature_name)


    _feature_names = list(feature_name.keys())
    _feature_types = list(feature_name.values())
     
    train_dataloader,eval_dataloader,test_dataload=None,None,None
    if rank_size>1:
        rank_id=get_rank()
        
        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle, num_shards=rank_size, shard_id=rank_id).batch(batch_size,drop_remainder=True)
        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size,drop_remainder=True)
        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size,drop_remainder=True)
    else:
        #单卡
        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle).batch(batch_size,drop_remainder=True)
        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size,drop_remainder=True)
        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size,drop_remainder=True)
    

    return train_dataloader, eval_dataloader, test_dataloader



def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True,rank_size=1):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len(dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if os.getenv('RANK_SIZE'):
        rank_size=int(os.getenv('RANK_SIZE'))
        
    train_dataset = padding(pad_item, pad_max_len, feature_name, data=train_data)
    eval_dataset = padding(pad_item, pad_max_len, feature_name, data=eval_data)
    test_dataset = padding(pad_item, pad_max_len, feature_name, data=test_data)
    
    train_dataset = pd.DataFrame(train_dataset)
    eval_dataset = pd.DataFrame(eval_dataset)
    test_dataset = pd.DataFrame(test_dataset)
    
    
    train_dataset.column_names=feature_name
    eval_dataset.column_names=feature_name
    test_dataset.column_names=feature_name
    
    train_dataset=dict(train_dataset)
    eval_dataset=dict(eval_dataset)
    test_dataset=dict(test_dataset)



    _feature_names = list(feature_name.keys())

    if "current_loc" in pad_item:
        _feature_names.append('loc_len')
        _feature_names.append('his_len')
    
    _feature_types = list(feature_name.values())
    
    
    

    train_dataloader,eval_dataloader,test_dataload=None,None,None
    if rank_size>1:
        rank_id=get_rank()
        
        train_dataloader = ds.NumpySlicesDataset(data=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle, num_shards=rank_size, shard_id=rank_id).batch(batch_size,drop_remainder=True)
        eval_dataloader = ds.NumpySlicesDataset(data=eval_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size,drop_remainder=True)
        test_dataloader = ds.NumpySlicesDataset(data=test_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size,drop_remainder=True)
    else:
        #单卡
        train_dataloader = ds.NumpySlicesDataset(data=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle).batch(batch_size,drop_remainder=True)
        eval_dataloader = ds.NumpySlicesDataset(data=eval_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size,drop_remainder=True)
        test_dataloader = ds.NumpySlicesDataset(data=test_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size,drop_remainder=True)
    return train_dataloader, eval_dataloader, test_dataloader
