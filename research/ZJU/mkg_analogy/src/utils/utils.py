import numpy as np
import mindspore.nn as nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR,\
    WarmUpLR, CosineDecayLR
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.metrics import Metric, rearrange_inputs

class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr

class AnalogyMetric(Metric):
    def __init__(self):
        """调用super进行初始化"""
        super(AnalogyMetric, self).__init__()
        self.clear()

    def clear(self):
        """清除内部计算结果，变量初始化"""
        self._entity_ranks = []
        self._rel_ranks = []

    @rearrange_inputs
    def update(self, *inputs):
        """更新内部计算结果"""
        # inputs: (ent_rank, rel_rank)
        entity_ranks, relation_ranks = inputs
        # 参数计算
        if entity_ranks is not None: 
            self._entity_ranks.append(entity_ranks)
        if relation_ranks is not None: 
            self._rel_ranks.append(relation_ranks)

    def eval(self):
        ent_hits10, ent_hit1 = None, None
        if len(self._entity_ranks) > 0:
            self._entity_ranks = np.concatenate(self._entity_ranks)
            # entity
            hits20 = (self._entity_ranks<=20).mean()
            hits10 = (self._entity_ranks<=10).mean()
            hits5 = (self._entity_ranks<=5).mean()
            hits3 = (self._entity_ranks<=3).mean()
            hits1 = (self._entity_ranks<=1).mean()

            ("Eval_entity/hits1", hits1)
            print("Eval_entity/hits3", hits3)
            print("Eval_entity/hits5", hits5)
            print("Eval_entity/hits10", hits10)
            print("Eval_entity/hits20", hits20)
            print("Eval_entity/mean_rank", self._entity_ranks.mean())
            print("Eval_entity/mrr", (1. / self._entity_ranks).mean())
            print("entity_hits10", hits10)
            print("entity_hits1", hits1)
            ent_hits10, ent_hit1 = hits10, hits1

        return ent_hits10, ent_hit1