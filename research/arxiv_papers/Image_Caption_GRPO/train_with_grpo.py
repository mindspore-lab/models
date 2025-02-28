import os
import copy
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
from module.utils import _sample

def set_seed(seed):
    random.seed(seed)  # 配置Python random库的随机种子
    np.random.seed(seed)  # 配置Numpy库的随机种子
    ms.set_seed(seed)  # 配置MindSpore库的随机种子

def make_experience(model, img, caption_index, config, img_id, train_dict):
    bs = img.shape[0]
    eos_index = ms.tensor([config.eos_token_id])

    # 生成sample_nums个回答
    img = img.unsqueeze(0).repeat_interleave(config.sample_nums, 0).reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    caption_index = caption_index.unsqueeze(0).repeat_interleave(config.sample_nums, 0).reshape(-1, caption_index.shape[-1])

    sample_index = _sample(model, img, caption_index, config)
    if sample_index.shape[1] < config.max_length:
        add_eos = ops.full([sample_index.shape[0], config.max_length - sample_index.shape[1]], fill_value = eos_index[0], dtype = sample_index.dtype)
        sample_index = ops.cat([sample_index, add_eos], axis = 1)
    assert sample_index.shape[1] == config.max_length

    # 计算生成时的概率
    pred = model(img, sample_index)
    logits = ops.log_softmax(pred, axis = -1)

    # 取出采样概率
    get_sample_index = ops.full_like(sample_index, config.eos_token_id)
    get_sample_index[:, :-1] = sample_index[:, 1:]
    logits = ops.gather_elements(logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
    logits = logits * (sample_index != config.eos_token_id).to(ms.float32)

    img = img.reshape(config.sample_nums, bs, img.shape[-3], img.shape[-2], img.shape[-1])
    sample_index = sample_index.reshape(config.sample_nums, bs, sample_index.shape[-1])
    logits = logits.reshape(config.sample_nums, bs, logits.shape[-1])

    all_reward_score = []
    for i in range(config.sample_nums):
        sample_pred_str = config.tokenizer.batch_decode(sample_index[i].reshape(bs, -1).tolist(), skip_special_tokens=True)

        gts = {}
        sample_res = {}
        for k in range(bs):
            image_id = int(img_id[k])
            gts[image_id] = train_dict.imgid_to_sentences[image_id]
            sample_res[image_id] = [sample_pred_str[k]]

        reward_score = compute_scores_cider(gts, sample_res)[1]['CIDEr']
        reward_score = ms.tensor(reward_score, dtype = ms.float32)
        all_reward_score.append(reward_score)

    all_reward_score = ops.stack(all_reward_score, axis = 0)
    all_advantage = (all_reward_score - ops.mean(all_reward_score, axis = 0)) / (ops.pow(ops.sum(ops.pow(all_reward_score - ops.mean(all_reward_score, axis = 0), 2), dim = 0) / all_reward_score.shape[0], 0.5) + 1e-5)
    all_advantage = ops.where(ops.isfinite(all_advantage), all_advantage, ops.zeros_like(all_advantage))

    img = img.permute(1, 0, 2, 3, 4)
    sample_index = sample_index.permute(1, 0, 2)
    logits = logits.permute(1, 0, 2)
    all_reward_score = all_reward_score.permute(1, 0)
    all_advantage = all_advantage.permute(1, 0)

    experience = []
    for i in range(bs):
        temp = {'img': img[i], 'sample_index': sample_index[i], 'log_probs': logits[i], 
                'reward_score': all_reward_score[i], 'advantage': all_advantage[i]}
        experience.append(temp)
    return experience

def train(config):

    # 加载模型
    model = CaptionModel(config)
    param_dict = ms.load_checkpoint(os.path.join(config.model_save_path, config.ck))
    ms.load_param_into_net(model, param_dict)
    print(model)

    # 参考模型
    ref_model = copy.deepcopy(model)
    for p in ref_model.trainable_params():
        p.requires_grad = False

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
    optimizer = optim.Adam(params=model.trainable_params(), lr=config.grpo_lr, weight_decay=config.weight_decay)
    grpo_all_steps = math.ceil(len(train_dict) / config.grpo_batch_size)
    scheduler = cosine_schedule_with_warmup(optimizer, 0, config.grpo_all_epoch * config.grpo_epoch * grpo_all_steps)

    def forward_fn(img, sample_index, log_probs, advantage):
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        sample_index = sample_index.reshape(-1, sample_index.shape[-1])
        log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        advantage = advantage.reshape(-1)

        pred = model(img, sample_index)
        logits = ops.log_softmax(pred, axis = -1)

        # 计算参考模型的loss
        ref_pred = ref_model(img, sample_index)
        ref_logits = ops.log_softmax(ref_pred, axis = -1)

        # 取出采样概率
        get_sample_index = ops.full_like(sample_index, config.eos_token_id)
        get_sample_index[:, :-1] = sample_index[:, 1:]

        logits = ops.gather_elements(logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
        logits = logits * (sample_index != config.eos_token_id).to(ms.float32)

        ref_logits = ops.gather_elements(ref_logits, dim = -1, index = get_sample_index.unsqueeze(-1)).squeeze(-1)
        ref_logits = ref_logits * (sample_index != config.eos_token_id).to(ms.float32)
        kl_loss = ops.exp(ref_logits - logits) - (ref_logits - logits) - 1

        # 计算模型损失
        ratio = ops.exp(logits - log_probs)
        advantage = advantage.unsqueeze(1)
        grpo_loss1 = advantage * ratio
        grpo_loss2 = advantage * ops.clamp(ratio, 1.0 - config.policy_clip_eps, 1.0 + config.policy_clip_eps)
        padding_mask = (sample_index != config.eos_token_id).to(ms.float32)
        loss = -ops.sum((ops.minimum(grpo_loss1, grpo_loss2) - config.beta * kl_loss) * padding_mask) / ops.sum(padding_mask)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # 开始训练
    for epoch in range(config.grpo_all_epoch):

        experience_list = []

        for i, batch in enumerate(train_data):
            model.set_train(False)
            with ms._no_grad():
                img = batch['img']
                caption_index = batch['caption']
                begin_index = caption_index[:, 0, 0].unsqueeze(1)
                experience = make_experience(model, img, begin_index, config, batch['img_id'], train_dict)
                experience_list.extend(experience)

            if ((i + 1) % config.grpo_step == 0) or ((i + 1) == all_steps):
                # 对数据进行打乱操作
                random.shuffle(experience_list)
                img = ops.stack([temp['img'] for temp in experience_list], axis = 0)
                sample_index = ops.stack([temp['sample_index'] for temp in experience_list], axis = 0)
                log_probs = ops.stack([temp['log_probs'] for temp in experience_list], axis = 0)
                advantage = ops.stack([temp['advantage'] for temp in experience_list], axis = 0)

                for grpo_epoch in range(config.grpo_epoch):
                    print(scheduler.get_last_lr())
                    j = 0
                    while j < len(experience_list):
                        mini_bs_img = img[j:min(j + config.grpo_batch_size, len(experience_list))]
                        mini_bs_sample_index = sample_index[j:min(j + config.grpo_batch_size, len(experience_list))]
                        mini_bs_log_probs = log_probs[j:min(j + config.grpo_batch_size, len(experience_list))]
                        mini_bs_advantage = advantage[j:min(j + config.grpo_batch_size, len(experience_list))]
                        loss, grads = grad_fn(mini_bs_img, mini_bs_sample_index, mini_bs_log_probs, mini_bs_advantage)
                        optimizer(grads)
                        scheduler.step()
                        if j % (50 * config.grpo_batch_size) == 0:
                            print('j/batch: {}/{} | grpo_epoch/grpo_epochs: {}/{} | i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(j, len(experience_list), grpo_epoch, config.grpo_epoch, i, all_steps, epoch, config.grpo_all_epoch, loss.item()))
                        j = j + config.grpo_batch_size

                experience_list = []

            if ((i + 1) % (all_steps // config.grpo_save_frequency) == 0) or ((i + 1) == all_steps):
                ms.save_checkpoint(model, os.path.join(config.model_save_path, 'rl_epoch_{}_i_{}.pt'.format(epoch, i + 1)))
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