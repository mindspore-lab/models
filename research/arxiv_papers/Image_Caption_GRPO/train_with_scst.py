import os
import math
import random
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import dataset as ds
from mindspore.experimental import optim
from module.utils import cosine_schedule_with_warmup
from eval import eval
from config import Config
from model import CaptionModel
from read_file import build_data
from evaluation import compute_scores_cider

# 同时进行贪心解码与采样解码, 并行运算加快速度
def greedy_and_sample(model, img, caption_index, config):
    bs = img.shape[0]
    caption_index = caption_index.unsqueeze(0).repeat_interleave(2, 0).reshape(-1, caption_index.shape[-1])
    isFin = ops.ones_like(caption_index).reshape(-1)
    isUnFin = ops.zeros_like(caption_index).reshape(-1)
    eos_index = ms.tensor([config.eos_token_id])

    # 计算一次图片编码, 加快解码速度
    img = img.unsqueeze(0).repeat_interleave(2, 0).reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    img_embed = model(img)
    for i in range(config.generation_length):

        # 若某个句子已经到达结束符, 将其状态设置为已完成
        last_token = caption_index[:, -1]
        flag = ops.where(last_token == eos_index, isFin, isUnFin)

        if ops.sum(flag) == ops.sum(isFin):
            break

        pred = model(img, caption_index, img_embed = img_embed)
        next = pred[:, -1, :]

        greedy_index, sample_index = ops.split(caption_index, bs, axis = 0)
        greedy_next, sample_next = ops.split(next, bs, axis = 0)
        greedy_flag, sample_flag = ops.split(flag, bs, axis = 0)

        # greedy解码部分
        greedy_next = ops.max(ops.log_softmax(greedy_next, axis = -1), axis = -1)

        # 若某个句子到达结束符, 只需要添加结束标签
        greedy_next_index = greedy_next[1]
        greedy_add_eos = ops.full_like(greedy_next_index, eos_index[0])
        greedy_next_index = ops.where(greedy_flag == 1, greedy_add_eos, greedy_next_index).reshape(-1, 1)
        greedy_index = ops.cat([greedy_index, greedy_next_index], axis = 1)

        # sample解码部分
        # 蒙特卡洛采样, 取Tok-k进行采样
        topk_values, topk_indices = ops.topk(sample_next, k = config.sample_top_k, dim = -1)
        normalized_probs = ops.softmax(topk_values, axis = -1)
        sampled_pos = ops.multinomial(normalized_probs, num_samples = 1)
        sample_next_index = ops.gather_elements(topk_indices, dim = 1, index = sampled_pos)

        # 若某个句子到达结束符, 只需要添加结束标签
        sample_next_index = sample_next_index.reshape(-1)
        sample_add_eos = ops.full_like(sample_next_index, eos_index[0])
        sample_next_index = ops.where(sample_flag == 1, sample_add_eos, sample_next_index).reshape(-1, 1)
        sample_index = ops.cat([sample_index, sample_next_index], axis = 1)

        caption_index = ops.cat([greedy_index, sample_index], axis = 0)

    greedy_index, sample_index = ops.split(caption_index, bs, axis = 0)
    return greedy_index, sample_index

def set_seed(seed):
    random.seed(seed)  # 配置Python random库的随机种子
    np.random.seed(seed)  # 配置Numpy库的随机种子
    ms.set_seed(seed)  # 配置MindSpore库的随机种子

def train(config):

    # 加载模型
    model = CaptionModel(config)
    param_dict = ms.load_checkpoint(os.path.join(config.model_save_path, config.ck))
    ms.load_param_into_net(model, param_dict)
    print(model)

    # 读取数据
    print("读取数据")
    ds.config.set_auto_offload(True)
    ds.config.set_enable_autotune(True)
    column_names = ['img', 'caption', 'label', 'img_id']
    train_dict = build_data(config)
    train_data = ds.GeneratorDataset(source = train_dict, column_names = column_names, shuffle = True)
    train_data = train_data.batch(config.batch_size)
    train_data = train_data.create_dict_iterator()

    configVal = Config(TrainOrVal = 'val')
    val_dict = build_data(configVal)
    val_data = ds.GeneratorDataset(source = val_dict, column_names = column_names, shuffle = False)
    val_data = val_data.batch(config.batch_size)
    val_data = val_data.create_dict_iterator()

    configTest = Config(TrainOrVal = 'test')
    test_dict = build_data(configTest)
    test_data = ds.GeneratorDataset(source = test_dict, column_names = column_names, shuffle = False)
    test_data = test_data.batch(config.batch_size)
    test_data = test_data.create_dict_iterator()

    print("train data is: ", len(train_dict))
    print("val data is: ", len(val_dict))
    print("test data is: ", len(test_dict))
    print("读取数据结束")

    all_steps = math.ceil(len(train_dict) / config.batch_size)
    optimizer = optim.Adam(params=model.trainable_params(), lr=config.rl_lr, weight_decay=config.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, 0, config.epoch * all_steps)

    def forward_fn(batch, sample_index, target_index):
        img = batch['img']
        pred = model(img, sample_index)
        logits = ops.log_softmax(pred, axis = -1)

        # 取出采样概率
        get_sample_index = ops.full_like(sample_index, config.eos_token_id)
        get_sample_index[:, :-1] = sample_index[:, 1:]
        logits = ops.gather_elements(logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
        logits = logits * (sample_index != config.eos_token_id).to(ms.float32)
        logits = ops.sum(logits, dim = -1)

        # 计算CIDEr
        sample_pred_str = config.tokenizer.batch_decode(sample_index.reshape(img.shape[0], -1).tolist(), skip_special_tokens=True)
        target_pred_str = config.tokenizer.batch_decode(target_index.reshape(img.shape[0], -1).tolist(), skip_special_tokens=True)

        gts = {}
        sample_res = {}
        greedy_res = {}
        bs = img.shape[0]
        for k in range(bs):
            image_id = int(batch['img_id'][k])
            gts[image_id] = train_dict.imgid_to_sentences[image_id]
            sample_res[image_id] = [sample_pred_str[k]]
            greedy_res[image_id] = [target_pred_str[k]]

        reward = compute_scores_cider(gts, sample_res)[1]['CIDEr']
        reward = ms.tensor(reward)

        baseline = compute_scores_cider(gts, greedy_res)[1]['CIDEr']
        baseline = ms.tensor(baseline)

        padding = (sample_index != config.eos_token_id).to(ms.float32)
        loss = -(reward - baseline) * logits
        loss = ops.sum(loss) / ops.sum(padding)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # 开始训练
    for epoch in range(config.epoch):
        print(scheduler.get_last_lr())
        for i, batch in enumerate(train_data):
            img = batch['img']
            caption_index = batch['caption']

            model.set_train(False)
            with ms._no_grad():
                begin_index = caption_index[:, 0, 0].unsqueeze(1)
                target_index, sample_index = greedy_and_sample(model, img, begin_index, config)

            loss, grads = grad_fn(batch, sample_index, target_index)
            optimizer(grads)
            scheduler.step()
            if i % 100 == 0:
                print('i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(i, all_steps, epoch, config.epoch, loss.item()))

        ms.save_checkpoint(model, os.path.join(config.model_save_path, 'rl_epoch_{}.ckpt'.format(epoch)))
        print("test:", end = ' ')
        with ms._no_grad():
            eval(configVal, model, val_data, val_dict)

    with ms._no_grad():
        eval(configTest, model, test_data, test_dict)

if __name__ == '__main__':
    ms.set_context(mode = ms.PYNATIVE_MODE, device_target = 'GPU')
    set_seed(Config().seed)
    config = Config()
    train(config)