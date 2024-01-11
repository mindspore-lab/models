import torch
import tqdm
import time
import os
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset.douban_domain_indicator import Douban, DoubanMusic, DoubanBook, DoubanMovie, DoubanMusic_sparse

from dataset.douban_split_v2 import DoubanMusic_split, DoubanMusic_sparse_split, DoubanBook_split, DoubanMovie_split

from model.dfm_embedding import DeepFactorizationMachineModel_embedding
from model.fnn_head import FactorizationSupportedNeuralNetworkModel_head
from denoising_diffusion.denoising_diffusion_1d_v2 import Unet1D, GaussianDiffusion1D, Unet1D_2, Unet1D_3


def get_dataset(name, path):
    if name == 'douban':
        return Douban()
    elif name == 'douban_music':
        #return DoubanMusic(path)
        return DoubanMusic_sparse(path)
    elif name == 'douban_book':
        return DoubanBook(path)
    elif name == 'douban_movie':
        return DoubanMovie(path)
    else:
        raise ValueError('unknown dataset name: ' + name)
        
def get_dataset_split(name, path, y):
    if name == 'douban':
        return Douban()
    elif name == 'douban_music':
        #return DoubanMusic_split(path,y)
        return DoubanMusic_sparse_split(path,y)
    elif name == 'douban_book':
        return DoubanBook_split(path,y)
    elif name == 'douban_movie':
        return DoubanMovie_split(path,y)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset, numerical_num = 0,expert_num=8, embed_dim=16):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    
    task_num = 3

    if name == 'fnn_head':
        return FactorizationSupportedNeuralNetworkModel_head(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'dfm_embedding':
        return DeepFactorizationMachineModel_embedding(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)
        
class EarlyStopper():

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = -np.inf
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            ms.save_checkpoint(model, self.save_path)
            return True
        if self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        return False

def train(model, optimizer, data_loader, criterion, log_interval=100):
    model.set_train(True)
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits
    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    for i, (fields, target) in enumerate(tk0):
        (loss, _), grads = grad_fn(fields, target)
        optimizer(grads)
        total_loss += sum(loss)
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader):
    model.set_train(False)
    targets, predicts = list(), list()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        y = model(fields)
        targets.extend(target.asnumpy().tolist())
        predicts.extend(y.asnumpy().tolist())
    return roc_auc_score(targets, predicts)

def main(dataset_name,
         dataset_path,
         model_name,
         mode,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         tem,
         device,
         save_dir,
         freeze,
         job,
         indexx,
         M,
         T,
         beta,
         schedule,
         objective,
         auto_normalize):
    ms.set_context(device_target=device)
    dataset_name = ['douban_music','douban_book','douban_movie']
    dataset_name=dataset_name[indexx]
    
    mode='train'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    
    train_dataset_0 = get_dataset_split(dataset_name, dataset_path, y=0)
    train_dataset_1 = get_dataset_split(dataset_name, dataset_path, y=1)
    
    mode='val'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    val_dataset_0 = get_dataset_split(dataset_name, dataset_path, y=0)
    val_dataset_1 = get_dataset_split(dataset_name, dataset_path, y=1)
    
    mode='test'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    test_dataset_0 = get_dataset_split(dataset_name, dataset_path, y=0)
    test_dataset_1 = get_dataset_split(dataset_name, dataset_path, y=1)
    
    train_data_0_loader = ms.dataset.GeneratorDataset(train_dataset_0, column_names=["item", "target"], shuffle=True)
    train_data_0_loader = train_data_loader.batch(batch_size, drop_remainder=False)
    valid_data_0_loader = ms.dataset.GeneratorDataset(valid_dataset_0, column_names=["item", "target"], shuffle=False)
    valid_data_0_loader = valid_data_loader.batch(batch_size, drop_remainder=False)
    test_data_0_loader = ms.dataset.GeneratorDataset(test_dataset_0, column_names=["item", "target"], shuffle=False)
    test_data_0_loader = test_data_loader.batch(batch_size, drop_remainder=False)

    train_data_1_loader = ms.dataset.GeneratorDataset(train_dataset_1, column_names=["item", "target"], shuffle=True)
    train_data_1_loader = train_data_loader.batch(batch_size, drop_remainder=False)
    valid_data_1_loader = ms.dataset.GeneratorDataset(valid_dataset_1, column_names=["item", "target"], shuffle=False)
    valid_data_1_loader = valid_data_loader.batch(batch_size, drop_remainder=False)
    test_data_1_loader = ms.dataset.GeneratorDataset(test_dataset_1, column_names=["item", "target"], shuffle=False)
    test_data_1_loader = test_data_loader.batch(batch_size, drop_remainder=False)
    
    save_path=f'{save_dir}/douban_{model_name}_train_v2_6.ckpt'

    print(objective)
    print(auto_normalize)
    print(schedule)
    print(beta)
    print(T)
    
    D = 16   # input dimension
    
    net0 = Unet1D_3(
        dim = D,
        dim_mults = (1, 2, 4, 8),
        #dim_mults = (2, 2),
        channels = 2
    )

    model0 = GaussianDiffusion1D(
        net0,
        seq_length = D,
        timesteps = T,
        beta_schedule = schedule,
        objective = objective,
        constant=beta,
        auto_normalize=auto_normalize
    )
    
    net1 = Unet1D_3(
        dim = D,
        dim_mults = (1, 2, 4, 8),
        #dim_mults = (2, 2),
        channels = 2
    )

    model1 = GaussianDiffusion1D(
        net1,
        seq_length = D,
        timesteps = T,
        beta_schedule = schedule,
        objective = objective,
        constant=beta,
        auto_normalize=auto_normalize
    )
    
    
    for name, param in model0.named_parameters():
        if ('embedding' in name):
            for name1, param1 in model_base.named_parameters():
                if (name1==name):
                    param1=param1.cpu()
                    param.data = param1.data
                    param.requires_grad = False
                    
    
    for name, param in model1.named_parameters():
        if ('embedding' in name):
            for name1, param1 in model_base.named_parameters():
                if (name1==name):
                    param.data = param1.data
                    param.requires_grad = False
    
    save_path=f'{save_dir}/{model_name}_{dataset_name}_diff1_{learning_rate}_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_{job}.ckpt'
    optimizer = torch.optim.Adam(params=model1.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    
    start = time.time()
    
    for epoch_i in range(epoch):
        train(model1, optimizer, train_data_1_loader, device)
        loss = test(model1, val_data_1_loader, device)
        print('epoch:', epoch_i, 'validation loss:', loss)
        if not early_stopper.is_continuable(model1, -loss):
            l=-early_stopper.best_accuracy
            print(f'validation best loss: {l}')
            break
            
    end = time.time()
    
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model1, param_dict)
    loss = test(model1, test_data_1_loader, device)
    print(f'test loss: {loss}')
    print('running time = ',end - start)
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='douban_music')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--model_name', default='mmoe')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tem', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0',help='cpu, cuda:0')
    parser.add_argument('--save_dir', default='/chkpt/')
    parser.add_argument('--freeze', type=int, default=5)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--indexx', type=int, default=0)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta', type=float, default=0.0001)
    parser.add_argument('--schedule', default='linear')
    parser.add_argument('--objective', default='pred_x0')
    parser.add_argument('--auto_normalize', type=int, default=1)
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.mode,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.tem,
         args.device,
         args.save_dir,
         args.freeze,
         args.job,
         args.indexx,
         args.M,
         args.T,
         args.beta,
         args.schedule,
         args.objective,
         args.auto_normalize)