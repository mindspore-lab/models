import mindspore as ms
import mindspore.ops as ops

def greedy_decode(model, img, caption_index, config):
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
        next = ops.max(ops.log_softmax(pred[:, -1, :], axis = -1), axis = -1)

        # 若某个句子到达结束符, 分数保持不变
        next_score = next[0]
        score_eos = ops.zeros_like(next_score)
        next_score = ops.where(flag == 1, score_eos, next_score)
        sum_log = sum_log + next_score

        # 若某个句子到达结束符, 只需要添加结束标签
        next_index = next[1]
        add_eos = ops.full_like(next_index, eos_index[0])
        next_index = ops.where(flag == 1, add_eos, next_index).reshape(-1, 1)
        caption_index = ops.cat([caption_index, next_index], axis = 1)

    return caption_index

def beam_decode(model, img, caption_index, config):
    bs = img.shape[0]

    # 计算一次图片编码, 加快解码速度
    img_embed = model(img)
    # 输入开始标签, 取得每个句子的第一个预测token, 每个句子将有beam_size个备选token, 同时得到预测分数
    out = model(img, caption_index, img_embed = img_embed)
    next = ops.log_softmax(out[:, -1, :], axis = -1)
    next = ops.topk(next, k = config.beam_size, dim = -1)

    sum_log = next[0].reshape(bs, -1)
    index = next[1].unsqueeze(2)

    # 开始拼接, 每个句子都有beam_size个可能的结果
    caption_index = caption_index.unsqueeze(1).repeat_interleave(config.beam_size, 1)
    caption_index = ops.cat([caption_index, index], axis = -1).reshape(bs, -1, caption_index.shape[-1] + 1)

    # 每张图片复制beam_size次
    img = img.unsqueeze(1).repeat_interleave(config.beam_size, 1).reshape(-1, img.shape[1], img.shape[2], img.shape[3])
    img_embed = model(img)

    # 判断每句话是否已完成解码
    isFin = ops.ones_like(sum_log)
    isUnFin = ops.zeros_like(sum_log)
    eos_index = ms.tensor([config.eos_token_id])

    for i in range(config.generation_length - 1):

        # 若某个备选句子已经到达结束符, 将其状态设置为已完成
        last_token = caption_index[:, :, -1]
        flag = ops.where(last_token == eos_index, isFin, isUnFin)

        # 若全部完成解码, 跳出循环
        if ops.sum(flag) == ops.sum(isFin):
            break

        flag = flag.reshape(-1, 1).unsqueeze(1).repeat_interleave(config.beam_size, 1).reshape(-1, 1)

        caption_index = caption_index.reshape(-1, caption_index.shape[-1])
        sum_log = sum_log.reshape(-1, 1)

        # 为每个备选再次生成beam_size个可能的答案
        out = model(img, caption_index, img_embed = img_embed)
        next = ops.log_softmax(out[:, -1, :], axis = -1)
        next = ops.topk(next, k = config.beam_size, dim = -1)

        # 计算每个句子所生成的beam_size * beam_size个候选句可能的得分, 若某个备选句flag为1
        # 则其生成的beam_size个候选句只选其中一个保留分数，其余beam_size - 1个候选句分数减去一个数
        # 这里减去的数只要大于0即可, 因为接下来的步骤为增添token, 当其状态已经结束时, 只需要不断添加结束符
        # 而其分数并不再改变, 因此若beam_size个候选句都被保留, 分数又都不变
        # 这个备选句生成的beam_size个候选句就有可能再接下来的一轮中全部成为备选句, 因此随机减去一个负值, 只保留一个分数即可
        next_score = next[0].reshape(-1, 1)
        score_eos = ms.tensor([0] + [-1] * (config.beam_size - 1)).to(ms.float32).unsqueeze(0).repeat_interleave(bs * config.beam_size, 0).reshape(-1, 1)
        next_score = ops.where(flag == 1, score_eos, next_score)
        temp_sum = sum_log.unsqueeze(1).repeat_interleave(config.beam_size, 1).reshape(-1, 1)
        temp_sum = (temp_sum + next_score).reshape(bs, -1)

        # 生成beam_size * beam_size个候选句可能的标签
        next_index = next[1].reshape(-1, 1)
        add_eos = ops.full_like(next_index, eos_index[0])
        next_index = ops.where(flag == 1, add_eos, next_index)
        temp_index = caption_index.unsqueeze(1).repeat_interleave(config.beam_size, 1).reshape(-1, caption_index.shape[-1])
        temp_index = ops.cat([temp_index, next_index], axis = 1).reshape(bs, -1, caption_index.shape[-1] + 1)

        # 从每个句子的beam_size * beam_size中选取beam_size个句子作为下一轮的备选句
        next = ops.topk(temp_sum, k = config.beam_size, dim = -1)
        sum_log = next[0]
        # caption_index = torch.stack([torch.index_select(temp_index[k], 0, next[1][k]) for k in range(bs)])
        caption_index = ops.gather_elements(temp_index, dim = 1, index = next[1].unsqueeze(2).repeat_interleave(temp_index.shape[-1], -1))

    # 从每个句子的beam_size个备选句中, 选择其中一个作为最终结果, 因为已经排序, 因此只需要取出beam_size中的第一个即可
    caption_index = caption_index[:, 0, :]

    return caption_index