import pickle
import os
import time
import shutil
import numpy as np
import yaml
from easydict import EasyDict
import src.data as data
from vocab import Vocabulary  # NOQA
# from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from ipdb import set_trace
import argparse
from mindspore import ops
import mindspore as ms
from mindspore import context
import mindspore

from src.pth2ckpt import pth2ckpt
from src.evaluation import i2t, AverageMeter, LogCollector, encode_data
from src.model import EncoderImage,EncoderText,BuildTrainNetwork,BuildValNetwork, CustomTrainOneStepCell,SimLoss
# max_length = 87

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='',
                        help='Config path.')
    args = parser.parse_args()
    with open(args.config) as f:
        opt0 = yaml.safe_load(f)
    opt = EasyDict(opt0['common'])
    # 创建文件夹
    if not os.path.exists(opt["logger_name"]):
        os.makedirs(opt["logger_name"])
    # 保存为yaml文件
    mpath = opt["logger_name"] + "/config.yaml"
    with open(mpath, "w") as f:
        yaml.dump(opt0, f, encoding='utf-8', allow_unicode=True)
    f.close()
    print(opt)

    context.set_context(device_id=opt.device_id, mode=context.GRAPH_MODE, device_target=opt.device_target)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    opt.distributed = False

    # 加载数据
    train_dataset, val_loader,train_dataset_len,val_dataset_len = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)
    print("train num：",train_dataset_len)
    print("test num：", val_dataset_len)

#     #定义模型
    grad_clip = opt.grad_clip
    img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                opt.finetune, opt.cnn_type,
                                no_imgnorm=opt.no_imgnorm,
                                self_attention=opt.self_attention)

    txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                               opt.embed_size, opt.num_layers,
                               no_txtnorm=opt.no_txtnorm,
                               self_attention=opt.self_attention,
                               embed_weights=opt.word_embed,
                               bi_gru=opt.bi_gru)

    #定义损失模型
    criterion = SimLoss(margin=opt.margin,
                             measure=opt.measure,
                             max_violation=opt.max_violation,
                             inner_dim=opt.embed_size)
    net_with_loss = BuildTrainNetwork(img_enc, txt_enc, criterion)
    
    

    
    valnet = BuildValNetwork(img_enc, txt_enc, criterion)

    save_all_model_dict =  {"net_image": img_enc,
                            "net_caption": txt_enc,
                            "criterion": criterion}

    # 封装模型损失、优化器
    params = list(txt_enc.trainable_params())
    params += list(img_enc.trainable_params())
    params += list(criterion.trainable_params())
    milestone = [18750//2, 37500//2, 56250//2]   #[18750, 37500, 56250]
    learning_rates = [opt.learning_rate, opt.learning_rate / 10, opt.learning_rate / 100]  
    output = ms.nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)

    optimizer = ms.nn.SGD(params, learning_rate=output, #opt.learning_rate,
                             momentum=opt.optimizer.momentum,
                             weight_decay=opt.optimizer.weight_decay,
                             nesterov=opt.optimizer.nesterov)
    train_net = CustomTrainOneStepCell(net_with_loss, optimizer)



    #转换预训练模型
    pre_model_path = opt.pre_model_path
    image_path = pre_model_path + "image.ckpt"
    ms.save_checkpoint(img_enc, image_path)
    text_path = pre_model_path + "text.ckpt"
    ms.save_checkpoint(txt_enc, text_path)
    criterion_path = pre_model_path + "criterion.ckpt"
    ms.save_checkpoint(criterion, criterion_path)
    model_path = pre_model_path + "checkpoint_110.pth.tar"
    save_path = pre_model_path + "cross_110weight_wanzheng.ckpt"
    pth2ckpt(model_path = model_path, save_path = save_path, image_path=image_path, text_path=text_path, criterion_path=criterion_path)

    state = ms.load_checkpoint(save_path)  #state是字典：{'name': name, 'data': param}
    #创建图像、文本、损失字典
    checkpoint_dict = {"net_image":{},"net_caption":{},"criterion":{}}
    for key, value in state.items():

        qianzu = key.split(".")[0]
        checkpoint_dict[qianzu][key] = value
    #加载
    ms.load_param_into_net(img_enc, checkpoint_dict["net_image"])
    ms.load_param_into_net(txt_enc, checkpoint_dict["net_caption"])
    ms.load_param_into_net(criterion, checkpoint_dict["criterion"])




    #加载模型
    
    if opt.resume:

        state = ms.load_checkpoint(opt.resume)  #state是字典：{'name': name, 'data': param}
        #创建图像、文本、损失字典
        checkpoint_dict = {"net_image":{},"net_caption":{},"criterion":{}}
        for key, value in state.items():

            qianzu = key.split(".")[0]
            checkpoint_dict[qianzu][key] = value
        #加载
        ms.load_param_into_net(img_enc, checkpoint_dict["net_image"])
        ms.load_param_into_net(txt_enc, checkpoint_dict["net_caption"])
        ms.load_param_into_net(criterion, checkpoint_dict["criterion"])
        print("load model success")   

    
    
    print("train star")
    steps = train_dataset.get_dataset_size()
    best_rsum = 0
    best_epoch = 0
    # 设置网络为训练模式

    zong_be_time = time.time()
    for epoch in range(opt.num_epochs):
        epoch_be_time = time.time()
        step = 0
        train_net.set_train()
        print("-----------------   " + "epoch " + str(epoch + 1) + "   -------------------")
        epoch_loss = 0
        for batch_data in train_dataset.create_dict_iterator():
            # set_trace()
            be_time = time.time()
            lengths = ops.Squeeze()(batch_data["lengths"])
            l_list = [int(i) for i in lengths.asnumpy().tolist()]
            mask_list = [ops.ExpandDims()(ms.Tensor(i * [0] + (opt.max_length + 3 - i) * [1], ms.int32), 0) for i in l_list]
            mask = ops.Concat(0)(mask_list)
            #模型训练
            result = train_net(batch_data["images"], batch_data["captions"], lengths, mask)
            epoch_loss += result 
            end_time = time.time()
            if step % 50 == 0:
                print(f"Epoch: [{epoch + 1} / {opt.num_epochs}], "
                    f"step: [{step} / {steps}], "
                    f"loss: {result}, "
                    f"time: {end_time-be_time}, ")
            step = step + 1
        print("epoch " + str(epoch + 1) +" 的loss:   "+ str(epoch_loss))
        
        val_be_time = time.time()
        rsum = validate(opt, val_loader, valnet, val_dataset_len, criterion)
        val_end_time = time.time()
        print("epoch " + str(epoch + 1) +" test time:   "+ str(val_end_time-val_be_time))
        epoch_end_time = time.time()
        print("epoch " + str(epoch + 1) +" train and test time:   "+ str(epoch_end_time-epoch_be_time))
        is_best = False
        if best_rsum < rsum:
            is_best = True
            best_rsum = rsum
            best_epoch = epoch+1
            #保存模型
            save_state_dict(save_all_model_dict, opt.logger_name ,epoch+1, is_best)
        print(" {} epoch val best, resum metrics: {}".format(best_epoch, best_rsum) )
    zong_end_time = time.time()
    print("train {} epoch time{}".format(opt.num_epochs, zong_end_time-zong_be_time))
    
        
        
        
            
            
def validate(opt, val_loader, model, val_dataset_len, criterion):
    # compute the encoding for all the validation images and captions
    print("start validate")

    img_embs, cap_embs, cap_masks = encode_data(
        model, val_loader, opt.log_step, print, val_dataset_len=val_dataset_len, opt = opt)

    # caption retrieval
    (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = i2t(img_embs, 
                                                                                                         cap_embs, 
                                                                                                         cap_masks,
                                                                                                         measure=opt.measure, 
                                                                                                         criterion=criterion,   #计算相似度矩阵
                                                                                                        opt=opt)
    # sum of recalls to be used for early stopping
    currscore = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr))
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr))
    print("rsum： ", currscore)
    return currscore

def load_state_dict(self, image_weight_path,text_weight_path):
    image_param_dict = load_checkpoint(image_weight_path)
    load_param_into_net(self.img_enc, image_param_dict)
    text_param_dict = load_checkpoint(text_weight_path)
    load_param_into_net(self.txt_enc, text_param_dict)

    
    
def model_save_multi(save_path: str,
                     models_dict: dict,
                     append_dict=None, print_save=True) -> None:
    """
    Input:
    save_path:模型保存的路径
    models_dict:多模型字典，例如：{'net_G': model,'net_D1': model_D1,'net_D2': model_D2}
    append_dict:附加信息字典，例如：{'iter': 10, 'lr': 0.01, 'Acc': 0.98}
    print_save:是否打印模型保存路径
    Output：
    None
    """
    params_list = []
    for model_name, model in models_dict.items():
        for name, param in model.parameters_and_names(name_prefix=model_name):
            params_list.append({'name': name, 'data': param})
    mindspore.save_checkpoint(params_list, save_path, append_dict=append_dict)
    if print_save:
        print('Save success , The model save path : {}'.format(save_path))    

def save_state_dict(save_all_model_dict, prefix,epoch, is_best=False):
    # state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    # return state_dict
    if not is_best:
        filename = prefix + '/checkpoint_{}.ckpt'.format(epoch)
        model_save_multi(filename, save_all_model_dict)
    else:
        filename = prefix + '/checkpoint_best.ckpt'
        model_save_multi(filename, save_all_model_dict)



if __name__ == '__main__':
    main()
