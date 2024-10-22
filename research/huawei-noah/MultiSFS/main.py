from mindspore import nn
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import tqdm
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import random
import time


from dataset import *
# from models.sharedbottom import SharedBottomModel
# from models.singletask import SingleTaskModel
# from models.omoe import OMoEModel
# from models.mmoe import MMoEModel, MMoEModel_pre
from models.ple import PLEModel
from models.aitm import AITMModel, ESMMModel
# from models.metaheac import MetaHeacModel
from pruner import *

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def get_dataset(name, path,task_t=2):
    if name == 'KuaiRand':
        return KuaiRand(path,task_t)
    elif 'AliCCP' in name:
        return AliCCP(path)
    elif name == 'QB':
        return Ten(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, categorical_field_dims, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'esmm':
        print("Model: ESMM")
        return ESMMModel(categorical_field_dims, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)

class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.auc=None
        self.logloss =None

    def is_continuable(self, model, auc,loss):
        accuracy = np.array(auc).mean()
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            self.auc = auc
            self.loss = loss
            ms.save_checkpoint(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, log_interval=100):
    def forward_fn(data,labels):
        y = model(data)
        loss_list = [criterion(y[i], labels[:, i]) for i in range(labels.shape[1])]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        return loss,y
    grad_fn = ops.value_and_grad(forward_fn,None,optimizer.parameters,has_aux=True)
    model.set_train(True)
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    tt = 0 
    for i, (categorical_fields, labels) in enumerate(loader):
        st = time.time()
        (loss,_), grads = grad_fn(categorical_fields, labels)
        tt += time.time()-st
        optimizer(grads)
        total_loss += loss
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    print(tt/(i+1))

def test(model, data_loader, task_num):
    bce = ops.BinaryCrossEntropy('none')
    model.set_train(False)
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    tt = 0
    ct = 0
    for categorical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        st = time.time()
        y = model(categorical_fields)
        tt += time.time()-st
        ct += 1
        for i in range(task_num):
            labels_dict[i].extend(labels[:, i].asnumpy().tolist())
            predicts_dict[i].extend(y[i].asnumpy().tolist())
            loss_dict[i].extend(bce(y[i], labels[:, i],None).asnumpy().tolist())
    print(tt/ct)
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results


def main(dataset_name,
         dataset_path,
         task_t,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir, 
         cr):
    ms.set_context(device_target=device) #CPU now
    if dataset_name == 'KuaiRand':
        train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv',task_t=task_t)
        test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv',task_t=task_t)
    else:
        train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv',task_t=task_t)
        test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv',task_t=task_t)
    task_num = task_t if task_t>2 else 2 # task_t: 0 for click and hate 1 for click and like 2 for like and comment, 3 for click like comment, 4 for all 
    field_dims = np.maximum(train_dataset.field_dims,test_dataset.field_dims)
    train_data_loader = ds.GeneratorDataset(train_dataset,column_names=["text", "label"], shuffle=True).batch(batch_size=batch_size, num_parallel_workers=4)
    test_data_loader = ds.GeneratorDataset(test_dataset,column_names=["text", "label"], shuffle=False).batch(batch_size=batch_size, num_parallel_workers=4)
    model = get_model(model_name, field_dims, task_num, expert_num, embed_dim)
    criterion = nn.BCELoss()
    cod = None
    if cr!=0:
        pruner = Prunner(model,criterion,train_data_loader)
        # rec = []
        # inds = np.meshgrid(np.arange(18,28),np.arange(18,28),np.arange(18,28))
        # cods = np.array(list(map(lambda x:x.ravel(),inds))).T
        # for cod in cods: 
        st=time.time()
        pruned_model, mask, idx = pruner.prun(compression_factor=cr,l=cod)
        print(time.time()-st)
        save_path=f'{save_dir}/{dataset_name}_{model_name}_pruned_{cr}.pt'
        optimizer = nn.Adam(params=pruned_model.trainable_params(), lr=learning_rate, weight_decay=weight_decay)
        early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
        for epoch_i in range(epoch):
            # if epoch_i == 3:
            #     pruned_model.embedding.embedding.weight.requires_grad = False
            train(pruned_model, optimizer, train_data_loader, criterion, device)
            auc, loss = test(pruned_model, test_data_loader, task_num, device)
            print('epoch:', epoch_i, 'test: auc:', auc)
            for i in range(task_num):
                print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
            if not early_stopper.is_continuable(pruned_model, auc, loss):
                print(f'test: best auc: {early_stopper.best_accuracy}')
                print(early_stopper.auc)
                print(early_stopper.loss)
                break
            #rec.append(cod.tolist()+auc)
            # with open(f'rec_per_{model_name}_{embed_dim}_{task_t}.pkl','wb') as f:
            #     pickle.dump(rec,f)
    else:
        save_path=f'{save_dir}/{dataset_name}_{model_name}.pt'
        optimizer = nn.Adam(params=model.trainable_params(), lr=learning_rate, weight_decay=weight_decay)
        early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
        for epoch_i in range(epoch):
            train(model, optimizer, train_data_loader, criterion, device)
            auc, loss = test(model, test_data_loader, task_num, device)
            print('epoch:', epoch_i, 'test: auc:', auc)
            for i in range(task_num):
                print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
            if not early_stopper.is_continuable(model, auc,loss):
                print(f'test: best auc: {early_stopper.best_accuracy}')
                print(early_stopper.auc)
                print(early_stopper.loss)
                break
    return early_stopper.auc, early_stopper.loss








        



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='KuaiRand', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US','KuaiRand','AliCCP1', 'AliCCP2', 'AliCCP3','QB'])
    parser.add_argument('--dataset_path', default='/home/yejinwang2/scratch/dataset/multi-task-data/')
    parser.add_argument('--model_name', default='mmoe', choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac', 'mmoe_pre','esmm'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_t', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--compress_ratio',type=float,default=0.5)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    set_random_seed(args.seed)
    rec = []
    for i in range(1):
        st=time.time()
        rec.append(main(args.dataset_name,
                args.dataset_path,
                args.task_t,
                args.expert_num,
                args.model_name,
                args.epoch,
                args.learning_rate,
                args.batch_size,
                args.embed_dim,
                args.weight_decay,
                args.device,
                args.save_dir,args.compress_ratio))
        print(time.time()-st)
    print(rec)
    print(np.array(rec).mean(0))
    # for i in range(3):
    #     seed = random.randint(0,9999)
    #     print(seed)
    # #seed = 42
    #     set_random_seed(seed)
    #     main(args.dataset_name,
    #             args.dataset_path,
    #             args.task_num,
    #             args.expert_num,
    #             args.model_name,
    #             args.epoch,
    #             args.learning_rate,
    #             args.batch_size,
    #             args.embed_dim,
    #             args.weight_decay,
    #             args.device,
    #             args.save_dir,args.compress_ratio)
