import math, copy
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.experimental import optim

def _sample(model, img, caption_index, config):
    isFin = ops.ones_like(caption_index).reshape(-1)
    isUnFin = ops.zeros_like(caption_index).reshape(-1)
    eos_index = ms.tensor([config.eos_token_id])

    sum_log = ops.zeros(img.shape[0])
    # 计算一次图片编码, 加快解码速度
    img_embed = model(img)
    for i in range(config.generation_length):

        # 若某个句子已经到达结束符, 将其状态设置为已完成
        last_token = caption_index[:, -1]
        flag = ops.where(last_token == eos_index, isFin, isUnFin)

        if ops.sum(flag) == ops.sum(isFin):
            break

        pred = model(img, caption_index, img_embed = img_embed)
        next = pred[:, -1, :]

        # 蒙特卡洛采样, 取Tok-k进行采样
        topk_values, topk_indices = ops.topk(next, k = config.sample_top_k, dim = -1)
        normalized_probs = ops.softmax(topk_values, axis = -1)
        sampled_pos = ops.multinomial(normalized_probs, num_samples=1)
        sample_index = ops.gather_elements(topk_indices, dim=1, index=sampled_pos)
        # 取出采样概率
        logits = ops.log_softmax(next, axis = -1)
        logits = ops.gather_elements(logits, dim = -1, index = sample_index)
        logits = logits.reshape(-1)

        # 若某个句子到达结束符, 分数保持不变
        score_eos = ops.zeros_like(logits)
        next_score = ops.where(flag == 1, score_eos, logits)
        sum_log = sum_log + next_score

        # 若某个句子到达结束符, 只需要添加结束标签
        sample_index = sample_index.reshape(-1)
        add_eos = ops.full_like(sample_index, eos_index[0])
        sample_index = ops.where(flag == 1, add_eos, sample_index).reshape(-1, 1)
        caption_index = ops.cat([caption_index, sample_index], axis = 1)

    return caption_index


class cosine_schedule_with_warmup(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.copy_lr = None
        super(cosine_schedule_with_warmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            self.copy_lr = copy.deepcopy(self._last_lr)
        if self.last_epoch < self.num_warmup_steps:
            # Warmup阶段：线性增加学习率
            scale = self.last_epoch / self.num_warmup_steps
            return [lr * scale for lr in self.copy_lr]
        else:
            # 余下的epochs使用余弦退火策略
            scale = max(0.0, 0.5 * (1. + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))))
            return [lr * scale for lr in self.copy_lr]


class Embeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(512, config.hidden_dim)

        self.LayerNorm = nn.LayerNorm((config.hidden_dim,))
        self.dropout = nn.Dropout(config.dropout)

    def construct(self, x):
        seq_length = x.shape[1]

        position_ids = ops.arange(seq_length, dtype=ms.int64)
        position_ids = position_ids.unsqueeze(0).repeat_interleave(x.shape[0], 0)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings