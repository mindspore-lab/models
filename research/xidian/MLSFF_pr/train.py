#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mindspore as ms
import tools.utils as utils
import time
from dataloader import  DataLoader
import eval_utils
import json
from nets.GSFF import GSFF
import opts                                  
from settings import input_json, vocab_dir
from tools.init import set_seed
import mindspore.ops as ops
#from mindspore import Profiler

def clip_grad(clip_value, grad):
    return ops.clip_by_value(grad, ops.scalar_to_tensor(-clip_value, grad.dtype),
                             ops.scalar_to_tensor(clip_value, grad.dtype))

class MyWithLossCell(ms.nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn
   def construct(self, inputs, labels):
       logits = self._backbone(inputs[:3],labels)
       loss = self._loss_fn(logits, labels[:,1:],inputs[3])
       return loss
   def backbone_network(self): 
       return self._backbone

args=opts.get_args()
vocab_file = json.load(open(vocab_dir,"r"))
args.num_attr = len(vocab_file)+1
set_seed(args.SEED)
loader = DataLoader(args)
args.vocab_size = loader.vocab_size  
print('vocab_size:',args.vocab_size) 
net=GSFF(args)                          
print('==> Using model unpretrained..')
best_val_metric=-1
epoch_start=0
all_batch_ind=0

if args.AdamW:
    print("using AdamW")
    epoch_lr = args.learning_rate
    optimizer=ms.nn.AdamWeightDecay(net.trainable_params(),epoch_lr,
                           eps=args.optim_epsilon,
                           weight_decay=args.weight_decay)        
else:
    print("using Adam")
    epoch_lr = args.learning_rate
    optimizer=ms.nn.Adam(net.trainable_params(),learning_rate=epoch_lr,
                           eps=args.optim_epsilon,
                           weight_decay=args.weight_decay)                    

crit = utils.LanguageModelCriterion()                                            
sc_flag = False
loss_net = MyWithLossCell(net, crit)
train_net = ms.nn.TrainOneStepCell(loss_net, optimizer)

def train(args):
    global all_batch_ind
    global best_val_metric     
    args.eval_every_steps=len(loader.split_ix['train'])//(args.batch_size)                                                     
    print('eval_every_steps:',args.eval_every_steps)
    try:
        for epoch in range(epoch_start,epoch_start+args.num_epoches):
            start=time.time()
            batch_idx=0
            while True:
                data = loader.get_batch('train')
                all_batch_ind=all_batch_ind+1
                batch_idx=batch_idx+1    
                tmp = [data['fc_feats'], data['att_feats'],  data['attr_feats'], data['labels'], data['masks']]
                tmp = [_ if _ is None else ms.Tensor(_) for _ in tmp]  
                fc_feats, att_feats, attr_feats ,labels, masks = tmp                  
                inputs = (fc_feats, att_feats, attr_feats, masks)
                train_net.set_train()
                train_net(inputs,labels)
                if all_batch_ind%1==0:
                    print('epoch:{},ind:{},tr_loss:{}'.format(epoch, all_batch_ind, loss_net(inputs, labels).asnumpy()))    
                if (all_batch_ind%(args.eval_every_iters)==0 and epoch>=0):
                    print('\n\n')
                    print('####################################################')
                    print('Testing...')                   
                    if  args.beam_size>1:
                        print('using beam search...',args.beam_size)                               
                    eval_kwargs=vars(args)
                    eval_kwargs['split']='eval'
                    eval_kwargs['dataset']= input_json
                    eval_kwargs['val_images_use']=len(loader.split_ix['eval'])                
                    _, _, lang_stats = eval_utils.eval_split(args,net, crit, loader, eval_kwargs)
                    print('Eval done!')
                    print('####################################################')
                    print('\n\n')
                    
                    global best_epoch 
                    if lang_stats['CIDEr']>=best_val_metric:
                        best_epoch=epoch
                        best_val_metric=lang_stats['CIDEr']
            
                    print('best  epoch is:{},the best val_metric is:{}'.format(best_epoch, best_val_metric))
                    print('\n\n\n')
                    end = time.time()
                    print('Current epoch cost {} minutes'.format(str( (end-start)/60)))           
                if data['bounds']['wrapped']:
                    break
        print('\n\n\n')
        print('best  epoch is:{},the best val_metric is:{}'.format(best_epoch, best_val_metric))

        
    except  KeyboardInterrupt:
        print('The program is terminated')

def main(args):
    train(args)
if __name__=='__main__':
    # PYNATIVE_MODE(1) or GRAPH_MODE(0)
    #profiler = Profiler()
    ms.set_context(mode=1, device_target="CPU")
    main(args)
    profiler.analyse()
