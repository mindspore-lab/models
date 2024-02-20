import mindspore as ms
import tqdm
import copy
import time
import os
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from dataset.douban import Douban,DoubanMusic_sparse,other
from dataset.douban_split_v2 import DoubanMusic_split, DoubanMusic_sparse_split, DoubanBook_split, DoubanMovie_split

from fnn_head import FactorizationSupportedNeuralNetworkModel_head
from dfm_embedding import DeepFactorizationMachineModel_embedding

from denoising_diffusion.denoising_diffusion_1d_v2 import Unet1D, GaussianDiffusion1D, classifier_2

def get_dataset(name, path=''):
    if name == 'douban':
        return Douban()
    elif name == 'douban_music':
        return DoubanMusic_sparse(path)
    else:
        return other('train')
        
def get_dataset_split(name, path, y):
    if name == 'douban':
        return Douban()
    elif name == 'douban_music':
        return DoubanMusic_sparse_split(path,y)
    elif name == 'douban_book':
        return DoubanBook_split(path,y)
    elif name == 'douban_movie':
        return DoubanMovie_split(path,y)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims

    if name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(256, 256,256), dropout=0)
    elif name == 'fnn_head':
        return FactorizationSupportedNeuralNetworkModel_head(field_dims, embed_dim=16, mlp_dims=(256, 256, 256), dropout=0)
    else:
        raise ValueError('unknown model name: ' + name)
        
class EarlyStopper(object):

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
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, model0, model1, model_emb, net, train_dataset_1, T, step1, step2, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields = fields.to(device)
        target = target.to(device)
        
        fields[:,0]=0
        
        fields=model_emb(fields)
        d_id=fields[0,0,:]
        
        #divide into y=0 and y=1
        x0=fields[target == 0,1:,:]
        x1=fields[target == 1,1:,:]
        
        #step 1
        #for y=0
        flag_0=mindspore.ops.zeros(x0.shape[0])
        
        dictionary_t0=mindspore.ops.zeros(x0.shape[0])
        noise = mindspore.ops.randn_like(x0)
        
        for k in range(step1,step2,1):
            if sum(flag_0==0)==0:
                break
            x_start = x0[flag_0==0]
            t = mindspore.ops.full((sum(flag_0==0),), k, device=device, dtype=mindspore.long)
            temp_flag = mindspore.ops.zeros(sum(flag_0==0))
            temp_dict = mindspore.ops.zeros(sum(flag_0==0))
            
            fields = model0.q_sample(x_start = x_start, t = t, noise=noise[flag_0==0]) #add noise
            predict = net(fields)
            if (sum(predict<0.5)==0):
                continue
            temp_flag[predict<0.5] = 1
            temp_dict[predict<0.5] = k
            dictionary_t0[flag_0==0] = temp_dict #dictionary_t0 set to be k
            flag_0[flag_0==0] = temp_flag #flag set to be 1
            
        
        #for y=1
        flag_1=mindspore.ops.zeros(x1.shape[0])
        dictionary_t1=mindspore.ops.zeros(x1.shape[0])
        noise = mindspore.ops.randn_like(x1)
        
        for k in range(step1,step2,1):
            if sum(flag_1==0)==0:
                break
            x_start = x1[flag_1==0]
            t = mindspore.ops.full((sum(flag_1==0),), k, device=device, dtype=mindspore.long)
            temp_flag = mindspore.ops.zeros(sum(flag_1==0))
            temp_dict = mindspore.ops.zeros(sum(flag_1==0))
            
            fields = model1.q_sample(x_start = x_start, t = t, noise=noise[flag_1==0]) #add noise
            predict = net(fields)
            if (sum(predict<0.5)==0):
                continue
            temp_flag[predict<0.5] = 1
            temp_dict[predict<0.5] = k
            dictionary_t1[flag_1==0] = temp_dict #dictionary_t0 set to be k
            flag_1[flag_1==0] = temp_flag #flag set to be 1
        
        #step 2
        flag_0[dictionary_t0<step1]=0 
        flag_1[dictionary_t1<step1]=0 
        
        x0=x0[flag_0==1]
        x1=x1[flag_1==1]
        dictionary_t0=dictionary_t0[flag_0==1]
        dictionary_t1=dictionary_t1[flag_1==1]
        #reverse
        diffsample_0 = model0.p_sample_loop_3(x0, dictionary_t0)
        diffsample_1 = model1.p_sample_loop_3(x1, dictionary_t1)
        
        #true data from other domains
        
        diff=mindspore.ops.cat((diffsample_0,diffsample_1),axis=0)
        domain_id= d_id.repeat(int(sum(flag_0))+int(sum(flag_1)),1,1)
        fields=mindspore.ops.cat((domain_id,diff),axis=1)
        target = mindspore.ops.cat((mindspore.ops.zeros(int(sum(flag_0))), mindspore.ops.ones(int(sum(flag_1)))), axis=0)
        
        fields, target = fields.to(device), target.to(device)
        
        #sparse true data 2048
        indexx = range(train_dataset_1.__len__())
        (fields_sparse_true, target_sparse_true) = train_dataset_1.__getitem__(indexx)
        fields_sparse_true=mindspore.Tensor(fields_sparse_true).to(device)
        fields_sparse_true=model_emb(fields_sparse_true)

        fields_sparse_true=fields_sparse_true.to(device)
        target_sparse_true=mindspore.Tensor(target_sparse_true).to(device)
        
        
        #sparse fake data 512
        indexxx = random.sample(range(train_dataset_1.__len__()), 512)
        (fields_sparse_fake, target_sparse_fake) = train_dataset_1.__getitem__(indexxx)
        
        diffsample_0 = model0.sample(batch_size = target_sparse_fake.shape[0]-int(sum(target_sparse_fake)))
        diffsample_1 = model1.sample(batch_size = int(sum(target_sparse_fake)))
        diff=mindspore.ops.cat((diffsample_0,diffsample_1),axis=0)
        domain_id= d_id.repeat(target_sparse_fake.shape[0],1,1)
        fields_sparse_fake=mindspore.ops.cat((domain_id,diff),axis=1)
        
        target_sparse_fake = mindspore.ops.cat((mindspore.ops.zeros(target_sparse_fake.shape[0]-int(sum(target_sparse_fake))), mindspore.ops.ones(int(sum(target_sparse_fake)))), axis=0)
        fields_sparse_fake=fields_sparse_fake.to(device)
        target_sparse_fake=target_sparse_fake.to(device)
        
        #concat
        
        fields=mindspore.ops.cat((fields, fields_sparse_true, fields_sparse_fake), axis=0)
        target=mindspore.ops.cat((target, target_sparse_true, target_sparse_fake), axis=0)
        
        #step 3, fine-tune
        
        y = model(fields)
        loss = criterion(y, target.float())
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device, model_emb):
    model.set_train(False)
    targets, predicts, loss = list(), list(), list()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        fields = fields.to(device)
        fields=model_emb(fields)
        target = target.to(device)
        
        fields = fields.to(device)
        
        y = model(fields)
        targets.extend(target.tolist())
        predicts.extend(y.tolist())
        loss.extend(mindspore.ops.binary_cross_entropy(y,target.float(), reduction='none').tolist())
    return -np.array(loss).mean()
        
def test_2(model, data_loader, device, model_emb):
    model.set_train(False)
    targets, predicts, loss = list(), list(), list()
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        fields = fields.to(device)
        fields=model_emb(fields)
        target = target.to(device)
        
        fields = fields.to(device)
        
        y = model(fields)
        targets.extend(target.tolist())
        predicts.extend(y.tolist())
        loss.extend(mindspore.ops.binary_cross_entropy(y,target.float(), reduction='none').tolist())
    return roc_auc_score(targets, predicts),np.array(loss).mean()

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
         auto_normalize,
         step1,
         step2):
    
    ms.set_context(device_target=device)
    name = ['douban_music','douban_book','douban_movie']
    dataset_name = ['douban_music','douban_book','douban_movie']
    del dataset_name[indexx]
    
    train_dataset=get_dataset('other')
    
    train_data_loader = ms.dataset.GeneratorDataset(train_dataset, column_names=["item", "target"], shuffle=True)
    train_data_loader = train_data_loader.batch(batch_size*10, drop_remainder=False)
    
    dataset_name=name[indexx]
    
    mode='train'
    dataset_1 = get_dataset(dataset_name,'/dataset/Douban/Data/douban_music/douban_music_sparse.csv')
    
    mode='train'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    train_dataset_1 = get_dataset(dataset_name, dataset_path)
    
    mode='val'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    valid_dataset_1 = get_dataset(dataset_name, dataset_path)
    
    mode='test'
    dataset_path='/dataset/Douban/Data/'+mode+'/'+dataset_name+'_sparse_'+mode+'.csv'
    test_dataset_1 = get_dataset(dataset_name, dataset_path)
    
    
    valid_data_loader = ms.dataset.GeneratorDataset(valid_dataset_1, column_names=["item", "target"], shuffle=False)
    valid_data_loader = valid_data_loader.batch(batch_size, drop_remainder=False)
    test_data_loader = ms.dataset.GeneratorDataset(test_dataset_1, column_names=["item", "target"], shuffle=False)
    test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)
    
    
    model_emb=get_model('dfm_embedding', dataset).to(device)
    rs=get_model(model_name+'_head', dataset).to(device)
    save_path=f'{save_dir}/{model_name}_{dataset_name}_train_6.pt'
    param_dict = ms.load_checkpoint(f'{save_dir}/douban_{model_name}_train_v2_6.pt')
    ms.load_param_into_net(model_base, param_dict)
    
    model_emb.embedding.embedding.load_state_dict(model_base.embedding.embedding.state_dict()) #key
    rs.mlp.mlp.load_state_dict(model_base.mlp.mlp.state_dict())

    D = 16   # input dimension
    
    save_path=f'{save_dir}/{model_name}_{dataset_name}_diff0_0.001_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_2.pt'
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model0, param_dict)
    save_path=f'{save_dir}/{model_name}_{dataset_name}_diff1_0.001_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_2.pt'
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model1, param_dict)
    
    for name, param in model0.named_parameters():
        param.requires_grad = False
    for name, param in model1.named_parameters():
        param.requires_grad = False
    
    model0 = model0.to(device)
    model1 = model1.to(device)
    model_emb = model_emb.to(device)
    
    #load the classifier
    save_path=f'{save_dir}/{model_name}_douban_classifier_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_2.pt'
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(net, param_dict)
    
    criterion = ms.nn.BCELoss()
    save_path=f'{save_dir}/{model_name}_{dataset_name}_diff_augment_final_v3_{learning_rate}_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_{step1}_{step2}_v2_{job}.pt'###################################
    optimizer = ms.nn.Adam(params=rs.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    
    start = time.time()
    for epoch_i in range(epoch):
        train(rs, optimizer, train_data_loader, criterion, device, model0, model1, model_emb, net, train_dataset_1, T, step1, step2)
        auc = test(rs, valid_data_loader, device, model_emb)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(rs, auc):
            
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
          
    end = time.time()
    
    rs=ms.load_checkpoint(save_path)
    auc, loss = test_2(rs, valid_data_loader, device, model_emb)
    print(f'valid auc: {auc}')
    print(f'valid logloss: {loss}')
    
    auc, loss = test_2(rs, test_data_loader, device, model_emb)
    print(f'test auc: {auc}')
    print(f'logloss: {loss}')
    print('running time = ',end - start)
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='douban_music')
    parser.add_argument('--dataset_path', default='/Douban/Data/')
    parser.add_argument('--model_name', default='fnn')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tem', type=float, default=1e-5)
    parser.add_argument('--device', default='GPU', help='CPU, GPU, Ascend, Davinci')
    parser.add_argument('--save_dir', default='/chkpt/')
    parser.add_argument('--freeze', type=int, default=5)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--indexx', type=int, default=0)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--beta', type=float, default=0.0002)
    parser.add_argument('--schedule', default='other')
    parser.add_argument('--objective', default='pred_v')
    parser.add_argument('--auto_normalize', type=int, default=1)
    parser.add_argument('--step1', type=int, default=100)
    parser.add_argument('--step2', type=int, default=100)
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
         args.auto_normalize,
         args.step1,
         args.step2)
