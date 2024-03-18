from mindspore.ops import functional as F
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
from ipdb import set_trace
import numpy as np
import mindspore.common.initializer as init
from math import sqrt
import mindspore.common.dtype as mstype
import mindspore 

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    # X:  (128, 36, 1024)
    # dim: -1
    norm = ops.Sqrt()(ops.Pow()(X, 2).sum(axis=dim, keepdims=True)) + eps   #(128, 36, 1)
    X = ops.Div()(X, norm)   #(128, 36, 1024)
    return X


class Softmax(nn.Cell):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.max = ops.ReduceMax(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.sub = ops.Sub()
        self.exp = ops.Exp()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()

    def construct(self, x):
        #变
        x = self.cast(x, mstype.float16)
        x = self.sub(x, self.max(x, self.axis))
        x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
        return x


def sum_attention(nnet, query, value, mask=None, dropout=None):

    scores = nnet(query).transpose(0,1,3,2)
    if mask is not None:

        mask = ops.Cast()(mask, ms.bool_)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = Softmax()(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    sum_attention_result = ops.matmul(p_attn, value)
    return sum_attention_result



class SummaryAttn(nn.Cell):

    def __init__(self, dim, num_attn, dropout, is_cat=False):
        super(SummaryAttn, self).__init__()
        self.linear = nn.SequentialCell([
                nn.Dense(dim, dim).to_float(mstype.float16),
                nn.ReLU(),
                nn.Dense(dim, num_attn).to_float(mstype.float16)
        ])
        self.h = num_attn
        self.is_cat = is_cat
        self.dropout = nn.Dropout(keep_prob = 1-dropout) if dropout > 0 else None
        self.expanddims = ops.ExpandDims()

    def construct(self, query, value, mask=None):
        if mask is not None:
            mask = self.expanddims(mask,-2)
        batch = query.shape[0]

        weighted = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout)
        weighted = weighted if self.is_cat else weighted.mean(axis=-2)

        return weighted





def qkv_attention(query, key, value, mask=None, dropout=None, transpose=None, sqrt32=None):
    """
    query   [5, 50, 50, 1024]
    key     [5, 50, 36, 1024]
    value   [5, 50, 36, 1024]
    mask    [50, 1, 50]
    """
    scores = None
    d_k = query.shape[-1]
    key = transpose(key, (0, 1, 3, 2))
    scores = ops.Div()(ops.BatchMatMul()(query, key), sqrt32) 
    p_attn = Softmax()(scores)
    qkv_attention_result = ops.matmul(p_attn, value)
    return qkv_attention_result




def qkv_attention_mask(query, key, value, mask=None, dropout=None, transpose=None, sqrt32 = None):
    """
    query   [5, 50, 50, 1024]
    key     [5, 50, 36, 1024]
    value   [5, 50, 36, 1024]
    mask    [50, 1, 50]
    """
    scores = None
    d_k = query.shape[-1]
    key = transpose(key, (0, 1, 3, 2))
    scores = ops.Div()(ops.BatchMatMul()(query, key), sqrt32)  #ops.Sqrt()(d_k)

    mask = ops.Cast()(mask, ms.bool_)
    scores = scores.masked_fill(mask, -1e9)

    p_attn = Softmax()(scores)
    qkv_attention_mask_result = ops.matmul(p_attn, value)
    return qkv_attention_mask_result



class GatedFusionNew(nn.Cell):
    def __init__(self, dim, num_attn, dropout=0.01, reduce_func="self_attn", fusion_func="concat"):
        super(GatedFusionNew, self).__init__()
        # fusion_func  concat
        self.dim = dim
        self.h = num_attn

        self.reduce_func = reduce_func
        self.fusion_func = fusion_func

        self.img_key_fc = nn.Dense(dim, dim, has_bias = False).to_float(mstype.float16)
        self.txt_key_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)

        self.img_query_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)
        self.txt_query_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)

        self.weighted_img_key_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)
        self.weighted_txt_key_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)

        self.weighted_img_query_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)
        self.weighted_txt_query_fc = nn.Dense(dim, dim, has_bias=False).to_float(mstype.float16)

        in_dim = 2 * dim


        self.fc_1 = nn.SequentialCell(
            [nn.Dense(in_dim, dim, has_bias=False).to_float(mstype.float16),
            nn.ReLU(),
            nn.Dropout(keep_prob=1-dropout)
            ])


        self.fc_2 = nn.SequentialCell([
            nn.Dense(in_dim, dim, has_bias=False).to_float(mstype.float16),
            nn.ReLU(),
            nn.Dropout(keep_prob=1-dropout)] )



        self.fc_out = nn.SequentialCell([
            nn.Dense(in_dim, dim).to_float(mstype.float16),
            nn.ReLU(),
            nn.Dropout(keep_prob=1-dropout),
            nn.Dense(dim, 1).to_float(mstype.float16),
            nn.Sigmoid()]
        )

        self.final_reduce_1 = SummaryAttn(dim, num_attn, dropout)
        self.final_reduce_2 = SummaryAttn(dim, num_attn, dropout)

        self.init_weights()

        self.sigmoid = ops.Sigmoid()
        self.expanddims = ops.ExpandDims()
        self.cat = ops.Concat(-1)
        self.transpose = ops.Transpose()
        print("GatedFusion module init success!")
        #变
        self.sqrt32 = ms.Tensor([32], ms.float16)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """

        r = np.sqrt(6.) / np.sqrt(self.dim +
                                  self.dim)
        self.img_key_fc.weight.set_data(init.initializer(
            init.Uniform(r), self.img_key_fc.weight.shape, self.img_key_fc.weight.dtype))
        self.txt_key_fc.weight.set_data(init.initializer(
            init.Uniform(r), self.txt_key_fc.weight.shape, self.txt_key_fc.weight.dtype))
        self.fc_1[0].weight.set_data(init.initializer(
            init.Uniform(r), self.fc_1[0].weight.shape, self.fc_1[0].weight.dtype))

        self.fc_2[0].weight.set_data(init.initializer(
            init.Uniform(r), self.fc_2[0].weight.shape, self.fc_2[0].weight.dtype))

        self.fc_out[0].weight.set_data(init.initializer(
            init.Uniform(r), self.fc_out[0].weight.shape, self.fc_out[0].weight.dtype))
        self.fc_out[0].bias.set_data(init.initializer(
            init.Constant(0), self.fc_out[0].bias.shape, self.fc_out[0].bias.dtype))

        self.fc_out[3].weight.set_data(init.initializer(
            init.Uniform(r), self.fc_out[3].weight.shape, self.fc_out[3].weight.dtype))
        self.fc_out[3].bias.set_data(init.initializer(
            init.Constant(0), self.fc_out[3].bias.shape, self.fc_out[3].bias.dtype))



    def construct(self, v1, v2, get_score=True, keep=None, mask=None):
        """
        v1         (1, 5, 36, 1024)
        v2         (200, 87, 1024)
        get_score
        keep
        mask       (200, 87)
        """
        expanddims = ops.ExpandDims()
        squeeze = ops.Squeeze(0)

        v1 = ops.Cast()(v1, ms.float16)
        v2 = ops.Cast()(v2, ms.float16)
        mask = ops.Cast()(mask, ms.float16)
        k1 = self.img_key_fc(v1)   #[5, 36, 1024]
        k2 = self.txt_key_fc(v2)   #[bath, 50文本词个数, 1024]
        q1 = self.img_query_fc(v1) #[5, 36, 1024]
        q2 = self.txt_query_fc(v2) #[bath, 50文本词个数, 1024]
        batch_size_v1 = v1.shape[0]
        batch_size_v2 = v2.shape[0]

        v1 = mindspore.numpy.tile(expanddims(v1, 1), (1, batch_size_v2, 1, 1))
        k1 = mindspore.numpy.tile(expanddims(k1, 1), (1, batch_size_v2, 1, 1))
        q1 = mindspore.numpy.tile(expanddims(q1, 1), (1, batch_size_v2, 1, 1))
        v2 = mindspore.numpy.tile(expanddims(v2, 0), (batch_size_v1, 1, 1, 1))
        k2 = mindspore.numpy.tile(expanddims(k2, 0), (batch_size_v1, 1, 1, 1))
        q2 = mindspore.numpy.tile(expanddims(q2, 0), (batch_size_v1, 1, 1, 1))

        weighted_v1 = qkv_attention(query=q2, key=k1, value=v1, transpose=self.transpose, sqrt32=self.sqrt32)  #正常
        weighted_v2 = qkv_attention_mask(q1, k2, v2,  self.expanddims(mask, 1), transpose=self.transpose, sqrt32=self.sqrt32)
        
        weighted_v2_q = self.weighted_txt_query_fc(weighted_v2)  #[5, 50, 36, 1024]
        weighted_v2_k = self.weighted_txt_key_fc(weighted_v2)

        weighted_v1_q = self.weighted_img_query_fc(weighted_v1)  #(5, 50, 50, 1024)
        weighted_v1_k = self.weighted_img_key_fc(weighted_v1)

        fused_v1 = qkv_attention(weighted_v2_q, weighted_v2_k, weighted_v2, transpose=self.transpose, sqrt32=self.sqrt32)
        fused_v2 = qkv_attention_mask(weighted_v1_q, weighted_v1_k, weighted_v1, self.expanddims(mask, -2), transpose=self.transpose, sqrt32=self.sqrt32)
         
            
        fused_v1 = l2norm(fused_v1)  #(5, 200, 36, 1024)
        fused_v2 = l2norm(fused_v2)

        gate_v1 = self.expanddims(self.sigmoid((v1 * fused_v1).sum(axis=-1)), -1)  #[5, 50, 36, 1]
        gate_v2 = self.expanddims(self.sigmoid((v2 * fused_v2).sum(axis=-1)), -1)

        co_v1 = self.cat((v1, fused_v1)) * gate_v1   #v1 (5, 200, 36, 1024)
        co_v2 = self.cat((v2, fused_v2)) * gate_v2
        co_v1 = self.fc_1(co_v1) + v1  #(5, 200, 36, 1024)
        co_v2 = self.fc_2(co_v2) + v2  #[5, 50, 50, 1024]
        
        co_v1 = self.final_reduce_1(co_v1, co_v1)
        co_v2 = self.final_reduce_2(co_v2, co_v2, mask)
        co_v1 = l2norm(co_v1)  #(5, 200, 1024)
        co_v2 = l2norm(co_v2)

        score = self.fc_out(self.cat((co_v1, co_v2)))  #(5, 200, 1)
        score = ops.Squeeze(-1)(score)  
        if keep:#== "regions":
            score = self.transpose(score, (1, 0))   #[50, 5]
        return score

