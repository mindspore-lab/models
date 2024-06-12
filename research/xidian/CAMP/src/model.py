
from src.rnns import GRU
from mindspore.ops import functional as F
from mindspore import load_checkpoint, load_param_into_net
import mindspore.numpy
import mindspore.ops as ops
from mindspore import nn
import mindspore as ms
import mindspore.common.initializer as init
from ipdb import set_trace
import numpy as np
from collections import OrderedDict
from time import *
import mindspore.common.dtype as mstype
from src.fusion_module import *
from ipdb import set_trace


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    # X:  (128, 36, 1024)
    # dim: -1
    norm = ops.Sqrt()(ops.Pow()(X, 2).sum(axis=dim, keepdims=True)) + eps   #(128, 36, 1)
    X = ops.Div()(X, norm)   #(128, 36, 1024)
    return X


class EncoderImage(nn.Cell):

    def __init__(self, data_name, img_dim, embed_size, finetune=False,
                 cnn_type='resnet152', no_imgnorm=False,
                 self_attention=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.self_attention = self_attention
        self.img_dim = img_dim

        self.fc = nn.Dense(img_dim, embed_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.img_dim + self.embed_size)
        self.fc.weight.set_data(init.initializer(
            init.Uniform(r), self.fc.weight.shape, self.fc.weight.dtype))
        self.fc.bias.set_data(init.initializer(
            init.Constant(0), self.fc.bias.shape, self.fc.bias.dtype))

    def construct(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features



#文本网络模型
class EncoderText(nn.Cell):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 bi_gru=False, no_txtnorm=False,
                 self_attention=False, embed_weights=''):
        super(EncoderText, self).__init__()
        self.no_txtnorm = no_txtnorm
        self.embed_size = embed_size
        self.self_attention = self_attention
        self.bi_gru = bi_gru

        # word embedding
        w_init = ms.common.initializer.Uniform(scale=0.1)
        self.embed = nn.Embedding(vocab_size, word_dim, embedding_table=w_init)


        # caption embedding
        self.rnn = GRU(word_dim,
                          embed_size,
                          num_layers,
                          has_bias=True,   #Whether the cell has bias b_ih and b_hh.
                          batch_first=True,   #the input and output tensors are provided as (batch, seq, feature)
                          bidirectional=True
                         )

    def construct(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        cap_emb, _ = self.rnn(x, seq_length=lengths)   #(128, 37, 2048)
        cap_emb = (cap_emb[:, :, :cap_emb.shape[2] // 2] + cap_emb[:, :, cap_emb.shape[2] // 2:]) / 2  #(128, 37, 1024)

        # normalization in the joint embedding space
        if not self.no_txtnorm:  #不执行
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb

#定义损失函数
class SimLoss(nn.Cell):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, inner_dim=0, loss_func="BCE"):
        super(SimLoss, self).__init__()
        self.margin = margin
        self.measure = measure  # gate_fusion_new
        self.sim = GatedFusionNew(inner_dim, 4, 0.0)

        self.loss_func = loss_func
        self.max_violation = max_violation
        self.eye = ops.Eye()
        self.log = ops.Log()
        self.expanddims = mindspore.ops.ExpandDims()
        self.drive_num = 1
        self.transpose = ops.Transpose()

    def construct(self, im, s, get_score=False, keep="words", mask=None):
        # compute image-sentence score matrix
        # keep             "regions"  测试执行这个
        # self.measure     'gate_fusion_new'
        
        """
        计算相似度矩阵(测试)
        cur_im   (1, 5, 36, 1024)
        cur_s    (200, 87, 1024)
        keep     'regions'
        mask     (200, 87)
        """
        scores = self.sim(v1=im, v2=s, keep=keep, mask=mask)
    
        if keep:# == "regions":
            # scores = scores.transpose(1, 0)
            scores = self.transpose(scores, (1, 0))

        if get_score:
            return scores

        eps = 0.000001

        scores = ms.ops.clip_by_value(scores, 0., 1.0 - eps)
        de_scores = 1.0 - scores
        eye_size = scores.shape[0]
        label = self.eye(eye_size, eye_size, ms.int32)
        de_label = 1 - label

        scores = self.log(scores) * label
        de_scores = self.log(de_scores) * de_label

        le = -(scores.sum() + scores.sum() + de_scores.min(1).sum() + de_scores.min(0).sum())

        return le



#定义损失模型
class BuildTrainNetwork(nn.Cell):
    def __init__(self, img_enc, txt_enc, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.net_image = img_enc
        self.net_caption = txt_enc
        self.criterion = criterion

    def construct(self, images, captions, lengths, mask):
        # images   (128, 36, 2048)
        # captions       (128,  n)
        # lengths        (128, 1)
        # ids            (128, 1)
        img_emb = self.net_image(images)
        cap_emb = self.net_caption(captions, lengths)
        loss = self.criterion(im=img_emb, s=cap_emb,mask=mask)
        return loss

#定义验证模型
class BuildValNetwork(nn.Cell):
    def __init__(self, img_enc, txt_enc, criterion):
        super(BuildValNetwork, self).__init__()
        self.net_image = img_enc
        self.net_caption = txt_enc
        self.criterion = criterion

    def construct(self, images, captions, lengths, mask):
        # images   (128, 36, 2048)
        # captions       (128,  n)
        # lengths        (128, 1)
        # ids            (128, 1)
        img_emb = self.net_image(images)
        cap_emb = self.net_caption(captions, lengths)
        return img_emb, cap_emb
        # img_emb   (128, 36, 1024)
        # cap_emb   (128, 87, 1024)
    

#封装损失网络和优化器
class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        loss = F.depend(loss, self.optimizer(grads))                   # 使用优化器更新权重参数
        return loss

