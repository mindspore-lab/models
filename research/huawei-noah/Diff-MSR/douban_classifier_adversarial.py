import mindspore as ms
import tqdm
import time
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from dataset.douban_mtl import Douban_mtl_classifier_binary
from dataset.douban_domain_indicator import Douban, DoubanMusic, DoubanBook, DoubanMovie

from model.dfm_embedding import DeepFactorizationMachineModel_embedding
from model.fnn_head import FactorizationSupportedNeuralNetworkModel_head

from denoising_diffusion.denoising_diffusion_1d_v2 import Unet1D, GaussianDiffusion1D, classifier, classifier_2, classifier_3

def get_dataset(name,mode='train'):
    if name == 'douban':
        return Douban_mtl_classifier_binary(mode)
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
        
class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
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

def train(model, optimizer, data_loader, criterion, device, model_emb, model0, step, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields= fields.to(device)
        fields=model_emb(fields)
        fields=fields[:,1:,:]
        fields, target = fields.to(device), target.to(device).long()
        field=fields
        x_start = fields
        noise = mindspore.ops.randn_like(x_start, device=device)
        t = mindspore.ops.randint(0, step, (fields.shape[0],), device=device).long()
        fields = model0.q_sample(x_start = x_start, t = t, noise=noise)
        fields=mindspore.ops.cat((field, fields), axis=0)
        y = model(fields)
        target=mindspore.ops.cat((target, target), axis=0)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device, model_emb, model0, step):
    model.set_train(False)
    targets, predicts = list(), list()
    num_correct = 0
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        fields= fields.to(device)
        fields=model_emb(fields)
        fields=fields[:,1:,:]
        fields, target = fields.to(device), target.to(device).long()
        field=fields
        
        x_start = fields
        noise = mindspore.ops.randn_like(x_start, device=device)
        t = mindspore.ops.randint(0, step, (fields.shape[0],), device=device).long()
        fields = model0.q_sample(x_start = x_start, t = t, noise=noise)
        fields=mindspore.ops.cat((field, fields), axis=0)
        y = model(fields)
       
        target=mindspore.ops.cat((target, target), axis=0)
        
        targets.extend(target.tolist())
        predicts.extend(y.tolist())
    
    return roc_auc_score(targets, predicts)
    
def test_2(model, data_loader, device, model_emb, model0, step):
    model.set_train(False)
    targets, predicts = list(), list()
    num_correct = 0
    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        fields= fields.to(device)
        fields=model_emb(fields)
        fields=fields[:,1:,:]
        fields, target = fields.to(device), target.to(device).long()
        field=fields
        
        x_start = fields
        noise = mindspore.ops.randn_like(x_start, device=device)
        t = mindspore.ops.randint(0, step, (fields.shape[0],), device=device).long()
        y = model(fields)
        
        targets.extend(target.tolist())
        predicts.extend(y.tolist())
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
         auto_normalize,
         step):
    
    ms.set_context(device_target=device)
    
    train_dataset = get_dataset(dataset_name,'train')
    valid_dataset = get_dataset(dataset_name,'val')
    test_dataset = get_dataset(dataset_name,'test')
    train_data_loader = ms.dataset.GeneratorDataset(train_dataset, column_names=["item", "target"], shuffle=True)
    train_data_loader = train_data_loader.batch(batch_size, drop_remainder=False)
    valid_data_loader = ms.dataset.GeneratorDataset(valid_dataset, column_names=["item", "target"], shuffle=False)
    valid_data_loader = valid_data_loader.batch(batch_size, drop_remainder=False)
    test_data_loader = ms.dataset.GeneratorDataset(test_dataset, column_names=["item", "target"], shuffle=False)
    test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)
    
    field_dims = train_dataset.field_dims
    model_emb=get_model('dfm_embedding', train_dataset).to(device)
    save_path=f'{save_dir}/douban_{model_name}_train_v2_6.pt'  
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model_base, param_dict)
    model_emb.embedding.embedding.load_state_dict(model_base.embedding.embedding.state_dict()) #key
    save_path=f'{save_dir}/{model_name}_douban_music_diff0_0.001_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_2.pt'
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(model0, param_dict)
    net=classifier_2(dim=16, channels = 2, embed_dims=(64,64)).to(device)
    
    criterion = ms.nn.BCELoss()
    save_path=f'{save_dir}/{model_name}_{dataset_name}_classifier_{T}_{beta}_{schedule}_{objective}_{auto_normalize}_v2_{job}.pt'
    optimizer = ms.nn.Adam(params=net.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(num_trials=10, save_path=save_path)
    
    start = time.time()
    
    for epoch_i in range(epoch):
        train(net, optimizer, train_data_loader, criterion, device, model_emb, model0, step)
        auc = test(net, valid_data_loader, device, model_emb, model0, step)
        print('epoch:', epoch_i, 'validation auc:', auc)
        if not early_stopper.is_continuable(net, auc):
            l=early_stopper.best_accuracy
            print(f'validation best auc: {l}')
            break
    end = time.time()
    
    param_dict = ms.load_checkpoint(save_path)
    ms.load_param_into_net(net, param_dict)
    auc = test(net, test_data_loader, device, model_emb, model0, step)
    print(f'test auc: {auc}')
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='douban')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--model_name', default='fnn')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tem', type=float, default=1e-5)
    parser.add_argument('--device', default='GPU', help='CPU, GPU, Ascend, Davinci')
    parser.add_argument('--save_dir', default='/chkpt/')
    parser.add_argument('--freeze', type=int, default=5)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--indexx', type=int, default=0)
    parser.add_argument('--M', type=int, default=64)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--schedule', default='other')
    parser.add_argument('--objective', default='pred_noise')
    parser.add_argument('--auto_normalize', type=int, default=0)
    parser.add_argument('--step', type=int, default=0)
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
         args.step)