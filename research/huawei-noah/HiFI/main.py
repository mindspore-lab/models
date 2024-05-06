from mindspore import nn
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import numpy as np
import os
import pandas as pd
import random
import time
import argparse

from dataset import HIFIDataset, EvalDataset, cal_js
from model import HIFIC
from model_item import HIFII

def argparser():
    parser = argparse.ArgumentParser()

    # path args
    parser.add_argument("--data_dir", type=str, default="data", help="data dir")
    parser.add_argument("--result_dir", type=str, default="logs", help="save dir")

    # data args
    parser.add_argument("--item_nums", type=int, default=100, help="item nums")
    parser.add_argument("--channel_nums", type=int, default=2, help="channel nums")
    parser.add_argument("--dense_dim", type=int, default=8, help="dense dim")
    parser.add_argument("--user_nums", type=int, default=100, help="user nums")
    parser.add_argument("--user_hist_len", type=int, default=10, help="user hist len")
    parser.add_argument("--seq_len", type=int, default=10, help="seq len")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="CPU")

    # train args
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch_num', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--soft_update_iter', type=int, default=10)
    parser.add_argument('--item_step_len', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)

    # model args
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--mlp_dim', type=int, default=16)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--use_rcpo', action='store_true')
    parser.add_argument('--crr_type', type=str, default='binary')
    parser.add_argument('--critic_weight', type=float, default=1.0)
    parser.add_argument('--use_gal', action='store_true')
    parser.add_argument('--num_heads', type=int, default=1)

    return parser.parse_args()

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def two_step_evaluate(params_dict, model_c, model_i, eval_dataloader):
    print("Start evaluating.")
    model_c.set_train(False)
    model_i.set_train(False)

    item_step_len = params_dict.get('item_step_len', 1)

    res_clicksum = []

    for i, train_batch_datas in enumerate(eval_dataloader):
        ranked_click_list = []
        cate_list = []

        cur_hist = train_batch_datas[0]
        cur_cand =train_batch_datas[1]
        cur_ranked = train_batch_datas[2]
        cur_pos = train_batch_datas[3]
        cur_js = train_batch_datas[4]
        cur_ch = train_batch_datas[5]
        click_0_list = train_batch_datas[6]
        click_1_list = train_batch_datas[7]
        batch_clicks = [click_0_list, click_1_list]

        pos_mask = ops.ones((cur_hist.shape[0], params_dict['seq_len']-1))
        pos_mask[:,7:]=0

        for time_step in range(params_dict['seq_len']-1):
            ranked_click_list.append([])
            cate_list.append([])

            pred_c, logits_c = model_c.infer([cur_hist,cur_cand,cur_ranked,cur_pos,cur_js,cur_ch])

            cur_ch[:, 0] = pred_c

            pred_i, logits_i = model_i.infer([cur_hist,cur_cand,cur_ranked,cur_pos,cur_js,cur_ch])
            real_pred = ops.argmax(logits_i*pos_mask, dim=-1)

            for idx in range(cur_hist.shape[0]):
                cur_c = int(pred_c[idx])
                cur_id = int(real_pred[idx])
                cur_c_sig = int(batch_clicks[cur_c][idx][cur_id])
                ranked_click_list[-1].append(cur_c_sig)
                cate_list[-1].append(cur_c)

                cur_ranked[idx,time_step,:] = cur_cand[idx,cur_c,cur_id,:]
                cur_cand[idx,cur_c,cur_id,:] = 0

                pos_mask[idx, cur_id] = 0
            cur_pos+=1
        out_click_list = np.array(ranked_click_list).transpose()
        # print(out_click_list.shape)
        cate_list = np.array(cate_list).transpose()
        # print(cate_list.shape)
        res_clicksum.extend(np.sum(out_click_list, axis=1).tolist())

    return np.mean(res_clicksum)


def soft_update_of_target_network(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.get_parameters(), local_model.get_parameters()):
        target_param.set_data(
            tau * local_param.value() + (1.0 - tau) * target_param.value()
        )

def train_one_epoch(params_dict, model_c, model_i, dataloader, optimizers, cur_epoch):
    def forward_fn(batch):
        all_states = []
        for j in range(6):
            all_states.append(batch[j])
        actions = batch[6]
        rewards = batch[8]
        dones = batch[10]
        loss, rcpo_loss, pred = model_c(all_states, actions, rewards, dones)
        return loss, rcpo_loss, pred

    def forward_fn_i(batch):
        all_states = []
        for j in range(6):
            all_states.append(batch[j])
        actions = batch[7]
        rewards = batch[9]
        dones = batch[10]
        loss, pred = model_i(all_states, actions, rewards, dones)
        return loss, pred

    print("Start training epoch:{}".format(cur_epoch))
    total_loss_c = 0
    total_loss_i = 0
    batch_cnt = 0

    grad_fn = ops.value_and_grad(forward_fn, None, optimizers[0].parameters, has_aux=True)
    grad_fn_i = ops.value_and_grad(forward_fn_i, None, optimizers[-1].parameters, has_aux=True)
    model_c.set_train(True)
    model_i.set_train(True)

    st = time.time()
    for i, train_batch_datas in enumerate(dataloader):
        batch_cnt+=1
        (loss_c, rcpo_grad, pred), grads = grad_fn(train_batch_datas)
        optimizers[0](grads)

        # print(model_c.actor_predict.mlp.trainable_params()[-2].asnumpy())
        (loss_i, pred_i), grads_i = grad_fn_i(train_batch_datas)
        optimizers[-1](grads_i)

        total_loss_i += loss_i
        total_loss_c += loss_c

        # update rcpo_lambda
        if params_dict['use_rcpo']:
            model_c.rcpo_lambda.set_data(model_c.rcpo_lambda.asnumpy() + params_dict['learning_rate'] * rcpo_grad)
            model_c.rcpo_lambda.set_data(np.clip(model_c.rcpo_lambda.asnumpy(), 0, 100))

        if i% params_dict['soft_update_iter'] == 0:
            # soft update target network
            soft_update_of_target_network(model_c.actor_predict, model_c.actor_target, params_dict['tau'])
            soft_update_of_target_network(model_c.critic_predict, model_c.critic_target, params_dict['tau'])
            if params_dict['use_rcpo']:
                soft_update_of_target_network(model_c.cost_predict, model_c.cost_target, params_dict['tau'])

            soft_update_of_target_network(model_i.actor_predict, model_i.actor_target, params_dict['tau'])
            soft_update_of_target_network(model_i.critic_predict, model_i.critic_target, params_dict['tau'])

    print("Epoch:{} loss_c:{} loss_i:{}. Time: {}s.".format(cur_epoch, total_loss_c/float(batch_cnt), total_loss_i/float(batch_cnt), time.time()-st))

def main(args):
    params_dict = vars(args)
    print("exp params:{}".format(params_dict))

    if not os.path.exists(params_dict['result_dir']):
        os.makedirs(params_dict['result_dir'])
    if not os.path.exists(params_dict['data_dir']):
        raise ValueError("Data dir not exists.")
    if not os.path.exists(os.path.join(params_dict['data_dir'],'inter_seq.csv')) \
        or not os.path.exists(os.path.join(params_dict['data_dir'],'item_feat.csv')) \
        or not os.path.exists(os.path.join(params_dict['data_dir'],'user_feat.csv')) \
        or not os.path.exists(os.path.join(params_dict['data_dir'],'test_inter_seq.csv')):
        raise ValueError("Data file not exists.")

    set_random_seed(params_dict['seed'])

    ms.set_context(device_target=params_dict['device'])  # CPU now

    train_dataset = HIFIDataset(data_dir=params_dict['data_dir'], params=params_dict)
    print("Finish loading data.")
    eval_dataset = EvalDataset(data_dir=params_dict['data_dir'], params=params_dict)

    train_data_loader = ds.GeneratorDataset(train_dataset, column_names=["s0","s1", "s2","s3", "s4","s5", "c_a", "i_a",
                        "c_r", "i_r", "done"], shuffle=False).batch(batch_size=params_dict['batch_size'])
    eval_data_loader = ds.GeneratorDataset(eval_dataset, column_names=["s0","s1", "s2","s3", "s4","s5", "c_0", "c_1"]
                                           , shuffle=False).batch(batch_size=3)
    model_c = HIFIC(params_dict)
    model_i = HIFII(params_dict)
    print("Finish init model.")

    if params_dict['use_rcpo']:
        ch_p = model_c.trainable_params()
        new_list = [x for x in ch_p if x.name != 'rcpo_lambda']
        ch_optimizer = nn.Adam(params=new_list, learning_rate=params_dict['learning_rate'])
        rcpo_optimizer = nn.Adam(params=[model_c.rcpo_lambda], learning_rate=params_dict['learning_rate'])
        all_optimizers = [ch_optimizer, rcpo_optimizer]
    else:
        all_optimizers = [nn.Adam(params=model_c.trainable_params(), learning_rate=params_dict['learning_rate'])]
    all_optimizers.append(nn.Adam(params=model_i.trainable_params(), learning_rate=params_dict['learning_rate']))

    best_metric = 0.
    for epoch_i in range(params_dict['epoch_num']):

        train_one_epoch(params_dict, model_c, model_i, train_data_loader, all_optimizers, epoch_i)

        if epoch_i% params_dict['eval_interval'] == 0:
            # evaluate
            sum_click_per_list = two_step_evaluate(params_dict, model_c, model_i, eval_data_loader)
            print(sum_click_per_list)

            if sum_click_per_list > best_metric:
                best_metric = sum_click_per_list
                print("Save best model in epoch {} with best metric {}.".format(epoch_i, best_metric))
                ms.save_checkpoint(model_c, os.path.join(params_dict['result_dir'], "best_model_c.ckpt"))
                ms.save_checkpoint(model_i, os.path.join(params_dict['result_dir'], "best_model_i.ckpt"))
                best_metric = sum_click_per_list

    print("Finish training.")



if __name__ == '__main__':
    args = argparser()
    print(args)
    main(args)