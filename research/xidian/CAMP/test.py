print("work start!")
# import tensorboard_logger as tb_logger
print("import logger OK!")
from src.model import EncoderImage,EncoderText,BuildTrainNetwork,BuildValNetwork, CustomTrainOneStepCell,SimLoss
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict
from ipdb import set_trace

from mindspore import load_checkpoint, load_param_into_net
import logging
import pickle
import os
from src.evaluation import i2t, AverageMeter, LogCollector, encode_data

import src.data as data
import src.model
from vocab import Vocabulary
import argparse
from src.fusion_module import *
import mindspore as ms
from mindspore import context
import mindspore
from ipdb import set_trace
from time import *


def test_CAMP_model(config_path, model_path):
    be_time = time()
    print("OK!")
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    
    with open(config_path) as f:
        opt = yaml.safe_load(f)  # ,Loader=yaml.Loader
    opt = EasyDict(opt['common'])

    opt.resume = model_path

    with open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    train_logger = LogCollector()

    context.set_context(device_id=opt.device_id, mode=context.GRAPH_MODE, device_target=opt.device_target)
    print("----Start init model----")
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
    testnet = BuildValNetwork(img_enc, txt_enc, criterion)

    # 加载模型
    
    if opt.resume:
        state = ms.load_checkpoint(opt.resume)  #state是字典：{'name': name, 'data': param}
        #创建图像、文本、损失字典
        checkpoint_dict = {"net_image":{},"net_caption":{},"criterion":{}}
        for key, value in state.items():
#             set_trace()
            qianzu = key.split(".")[0]
            checkpoint_dict[qianzu][key] = value
        #加载
        ms.load_param_into_net(img_enc, checkpoint_dict["net_image"])
        ms.load_param_into_net(txt_enc, checkpoint_dict["net_caption"])
        ms.load_param_into_net(criterion, checkpoint_dict["criterion"])
    print("完成加载模型")

    
    
    test_loader, test_loader_len = data.get_test_loader("test", 
                                                        opt.data_name, 
                                                        vocab, 
                                                        128, 
                                                        4, 
                                                        opt)
    """
    img_embs     (200, 36, 1024)
    cap_embs     (200, 87, 1024)
    cap_masks    (200, 87)
    """
    print("完成加载数据")

    
    
    img_embs, cap_embs, cap_masks = encode_data(model = testnet, 
                                                data_loader = test_loader,
                                                log_step = opt.log_step,
                                                logging = logging.info,
                                                val_dataset_len = test_loader_len,
                                                opt = opt)
    print("完成前向传播计算数据特征")

    
    
    (r1, r5, r10, medr, meanr), (r1i, r5i, r10i, medri, meanri), score_matrix = i2t(img_embs, 
                                                                                    cap_embs, 
                                                                                    cap_masks,
                                                                                    measure=opt.measure,
                                                                                    criterion=criterion,   #计算相似度矩阵
                                                                                    return_ranks=True,
                                                                                    opt=opt)
    resum = r1 + r5 + r10 + r1i + r5i + r10i
    print("完成指标计算")
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    end_time = time()
    print("测试总时间：",end_time-be_time)
    print("resum:\t{}".format(resum))
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model_path', default="./config.yaml",   #   ./data/
                        help='path to datasets')
    parser.add_argument('--test_config_path', default="./logs/checkpoint_best.ckpt",                                      #precomp
                        help='{coco,f30k}_precomp')
    opt_ = parser.parse_args()

    test_CAMP_model(opt_.test_config_path, model_path = opt_.test_model_path)



if __name__ == '__main__':
    main()
