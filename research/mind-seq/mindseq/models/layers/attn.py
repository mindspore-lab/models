import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import numpy as np

from ...utils.masking import TriangularCausalMask, ProbMask, ProbMaskCell
from ...utils.tools import mask_fill, get_attn_adj

###########################  JAT  ###############################
class JatAttn(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.jat_heads = config.jat_heads
        self.orders = config.order
        
        self.dropout = nn.Dropout(p=config.dropout)
        self.super_p = config.super_p
        self.prob_jat = config.prob_jat
        self.factor = config.factor_jat
        self.jat_sgn = config.jat_sgn
        self.device = config.device

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        index_sample = ops.randint(low=0, high=L_K, size=(L_Q, sample_k))
        index_sample[index_sample == L_K] = L_K - 1
        if self.device == "Ascend":
            K_samples = []
            for i in range(L_Q):
                K_samples.append(ops.gather(K, index_sample[i], axis=2).unsqueeze(2))
            K_sample = ops.cat(K_samples, axis=2)
        else:
            K_expand = ops.BroadcastTo(shape = (B, H, L_Q, L_K, E))(ops.expand_dims(K, -3))
            K_sample = K_expand[:, :, ops.expand_dims(ms.numpy.arange(L_Q), 1), index_sample, :]
        
        Q_K_sample = ops.matmul(Q.unsqueeze(-2), K_sample.swapaxes(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - ops.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        Q_reduce = Q[ops.arange(B)[:, None, None],
                     ops.arange(H)[None, :, None],
                     M_top, :]
        K_reduce = K[ops.arange(B)[:, None, None],
                     ops.arange(H)[None, :, None],
                     M_top, :]
        Q_K = ops.matmul(Q_reduce, K_reduce.swapaxes(-2, -1))
        return Q_K, M_top, Q_reduce, K_reduce
    
    def _merge(self, A, M_top, full_x):
        b, h, _, _ = A.shape
        resC = full_x
        resC[ops.arange(b)[:, None, None, None],
              ops.arange(h)[None, :, None, None],
              M_top.unsqueeze(-1),
              M_top.unsqueeze(-2)] = A
        return resC
    
    def construct(self, query, key):
        query = query[:,:,:self.jat_heads,:]
        key = key[:,:,:self.jat_heads,:]
        query = query.swapaxes(2, 1)
        key = key.swapaxes(2, 1)
        
        # x = ops.einsum("bhle,bhse->bhls", query, key)
        x = query.matmul(key.swapaxes(-2, -1))
        
        full_x = x
        B, H, Lk, L = x.shape
        if self.prob_jat:
            _, _, L_K, _ = key.shape
            _, _, L_Q, _ = query.shape
            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
            U_part = U_part if U_part<L_K else L_K
            u = u if u<L_Q else L_Q
            x, m_top, Q_reduce, K_reduce = self._prob_QK(query, key, sample_k=U_part, n_top=u)
        if self.prob_jat:
            A = get_attn_adj(x, K_reduce, self.super_p, self.jat_sgn)
        else:
            A = get_attn_adj(x, key, self.super_p, self.jat_sgn)
        
        out = []
        if self.prob_jat:
            x1 = ops.matmul(A, Q_reduce)
            x1 = ops.matmul(x1, K_reduce.swapaxes(-1, -2))
            x1 = ops.matmul(x1, A.swapaxes(-1, -2))
            out.append(self._merge(x1, m_top, full_x))
            for i in range(1, self.orders):
                x1 = ops.matmul(A, x1)
                x1 = ops.matmul(x1, A.swapaxes(-1, -2))
                out.append(self._merge(x1, m_top, full_x))
        else:
            x1 = ops.matmul(A, query)
            x1 = ops.matmul(x1, key.swapaxes(-1, -2))
            x1 = ops.matmul(x1, A.swapaxes(-1, -2))
            out.append(x1)
            for i in range(1, self.orders):
                x1 = ops.matmul(A, x1)
                x1 = ops.matmul(x1, A.swapaxes(-1, -2))
                out.append(x1)
        h = ms.Tensor([0.])
        for o in out:
            h = h + o[:, :, :Lk, :L]
        h = self.dropout(h)
        return h


class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, args=None):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)
        self.ceof = 0. if args is None else args.ceof

    def construct(self, queries, keys, values, attn_mask, jat_attn=None):
        B, L, H, E = queries.shape

        scores = ops.BatchMatMul()(queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 3, 1))
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)
            scores = mask_fill(attn_mask.mask, scores, -np.inf)
        if jat_attn is not None:
            B, jat_head, _, _ = jat_attn.shape
            scores[:, :jat_head, :, :] = scores[:, :jat_head, :, :] * (1 - self.ceof) + jat_attn * self.ceof
        if self.scale is None:
            A = self.dropout(ops.Softmax()((1./ops.sqrt(ms.Tensor(E, ms.float32))) * scores))
        else:
            A = self.dropout(ops.Softmax()(self.scale * scores))
        
        # A = self.dropout(ops.Softmax()(self.scale * scores))
        V = ops.BatchMatMul()(A, values.transpose((0, 2, 1, 3))).transpose(0, 2, 1, 3)
        return V

from mindspore.ops.function import broadcast_to
from mindspore import numpy as ms_np
import mindspore as ms
###########################  Prob  ###############################
class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, args=None):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)
        self.ceof = args.ceof
        self.device = args.device

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        index_sample = ops.UniformInt()((L_Q, sample_k), ms.Tensor(0, ms.int32), ms.Tensor(L_K, ms.int32)) # real U = U_part(factor*ln(L_k))*L_q
        index_sample[index_sample == L_K] = L_K - 1
        
        if self.device == "Ascend":
            K_samples = []
            for i in range(L_Q):
                K_samples.append(ops.gather(K, index_sample[i], axis=2).unsqueeze(2))
            K_sample = ops.cat(K_samples, axis=2)
        else:
            K_expand = ops.BroadcastTo(shape = (B, H, L_Q, L_K, E))(ops.expand_dims(K, -3))
            K_sample = K_expand[:, :, ops.expand_dims(ms.numpy.arange(L_Q), 1), index_sample, :]
        
        Q_K_sample = ops.Squeeze(-2)(ops.BatchMatMul()(ops.expand_dims(Q, -2), K_sample.swapaxes(-2, -1)))
        # find the Top_k query with sparisty measurement
        M = ops.ArgMaxWithValue(-1)(Q_K_sample)[0] - ops.div(ops.ReduceSum()(Q_K_sample, -1), L_K)
        M_top = ops.TopK(sorted=False)(M, n_top)[1]
         # use the reduced Q to calculate Q_K
        Q_reduce = Q[ms.numpy.arange(B)[:, None, None],
                     ms.numpy.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = ops.BatchMatMul()(Q_reduce, K.swapaxes(-2, -1)) # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = ops.ReduceMean()(V, -2)
            contex = ops.BroadcastTo(shape = (B, H, L_Q, V_sum.shape[-1]))(ops.expand_dims(V_sum, -2)).copy()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = ops.cumsum(V, -2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            _mask = ops.ones((L_Q, scores.shape[-1]), ms.bool_).triu(diagonal=1)
            # _mask = ops.triu(ops.ones((L_Q, scores.shape[-1]), ms.bool_), diagonal=1)
            _mask_ex = broadcast_to(_mask[None, None, :], (B, H, L_Q, scores.shape[-1]))
            indicator = _mask_ex[ms_np.arange(B)[:, None, None],
                                ms_np.arange(H)[None, :, None],
                                index, :]
            final_mask = indicator.view(scores.shape)

            scores = mask_fill(final_mask, scores, -np.inf)
            # attn_mask = ProbMask(B, H, L_Q, index, scores)
            # scores = mask_fill(attn_mask.mask, scores, -np.inf)
        attn = ops.Softmax()(scores)
        context_in[ms.numpy.arange(B)[:, None, None],
                   ms.numpy.arange(H)[None, :, None],
                   index, :] = ops.BatchMatMul()(attn, V).astype(context_in.dtype)
        
        attns = (ops.ones(((B, H, L_V, L_V)), ms.float32) / L_V).astype(attn.dtype)
        attns[ms.numpy.arange(B)[:, None, None], ms.numpy.arange(H)[None, :, None], index, :] = attn
        return context_in

    def construct(self, queries, keys, values, attn_mask, jat_attn=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.swapaxes(2, 1)
        keys = keys.swapaxes(2, 1)
        values = values.swapaxes(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        if self.scale is None:
            scores_top = scores_top * (1./ops.sqrt(ms.Tensor(D, ms.float32)))
        else:
            scores_top = scores_top * self.scale
        if jat_attn is not None:
            B, jat_head, _, _ = jat_attn.shape
            jat_attn = jat_attn[ops.arange(B)[:, None, None],
                     ops.arange(jat_head)[None, :, None],
                     index[:, :jat_head, :], :]
            scores_top[:, :jat_head, :, :] = scores_top[:, :jat_head, :, :] * (1 - self.ceof) + jat_attn * self.ceof
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.swapaxes(2, 1)
    
class moving_avg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def construct(self, x):
        # padding on the both ends of time series
        front = ops.tile(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = ops.tile(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.cat([front, x, end], axis=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False, jat_attention=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.jat_attention = jat_attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.avg = moving_avg(25, 1)

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        jat_queries = self.avg(queries)
        jat_keys = self.avg(keys)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        jat_queries = self.query_projection(jat_queries).view(B, L, H, -1)
        jat_keys = self.key_projection(jat_keys).view(B, S, H, -1)
        jat_attn = None

        if self.jat_attention is not None:
            jat_attn = self.jat_attention(
                jat_queries,
                jat_keys
            )

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            jat_attn,
        )
        if self.mix:
            out = out.swapaxes(2, 1)
        out = out.view(B, L, -1)
        return self.out_projection(out)

###########################  ALLOT  ###############################
import numpy as np
from math import sqrt
from ...utils.masking import ALLOTTriangularCausalMask
from .gcn import GraphConv
import mindspore as ms
import mindspore.nn as nn
def elu_feature_map(x):
    return ms.ops.elu(x) + 1

def bj(x, y, eps=1e-5):
    return (ms.ops.abs(x - y) > 1e-5).sum()

class STLinearAttention(nn.Cell):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STLinearAttention, self).__init__()
        self.feature_map = elu_feature_map # feature_map or elu_feature_map
        self.eps = scale or 1e-6
        
    def construct(self, queries, keys, values, attn_mask):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        KV = values.permute(0, 1, 3, 4, 2).matmul(K.permute(0, 1, 3, 2, 4)) #bnhms @ bnhsd
        # KV = ms.ops.einsum("bnshd,bnshm->bnhmd", K, values)

        # Z = 1/(ms.ops.einsum("bnlhd,bnhd->bnlh", Q, K.sum(axis=2))+self.eps)
        Z = 1/(ms.ops.mul(Q, K.sum(axis=2, keepdims=True)).sum(-1)+self.eps)
        
        # V = ms.ops.einsum("bnlhd,bnhmd,bnlh->bnlhm", Q, KV, Z)
        V = Q.unsqueeze(4).mul(KV.unsqueeze(2)).sum(-1).mul(Z.unsqueeze(-1))
        return V

class STFullAttention(nn.Cell):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(p=attention_dropout)

    def construct(self, queries, keys, values, attn_mask):
        B, N, L, H, E = queries.shape
        _, _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        # scores2 = ms.ops.einsum("bnlhd,bnshd->bnhls", queries, keys)
        scores = queries.swapaxes(2, 3).matmul(keys.permute(0, 1, 3, 4, 2))
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute the attention and the weighted average
        A = self.dropout(ms.ops.softmax(scale * scores, axis=-1))
        # V = ms.ops.einsum("bnhls,bnshd->bnlhd", A, values)
        V = A.swapaxes(2, 3).unsqueeze(4).mul(values.permute(0, 1, 3, 4, 2).unsqueeze(2)).sum(-1)

        return V


class STSLinearAttention(nn.Cell):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(STSLinearAttention, self).__init__()
        self.feature_map = elu_feature_map # feature_map or elu_feature_map
        self.eps = scale or 1e-6
        
    def construct(self, queries, keys, values, attn_mask):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # KV2 = ms.ops.einsum("bnhd,bnhm->bhmd", K, values)
        KV = K.permute(0, 2, 3, 1).matmul(values.permute(0, 2, 1, 3)).swapaxes(-1, -2)
        # Z2 = 1/(ms.ops.einsum("bnhd,bhd->bnh", Q, K.sum(axis=1))+self.eps)
        Z = 1/((Q.mul(K.sum(axis=1).unsqueeze(1)).sum(-1))+self.eps)
        # V2 = ms.ops.einsum("bnhd,bhmd,bnh->bnhm", Q, KV, Z)
        # V = ms.ops.einsum("bnhd,bhmd->bnhm", Q, KV)
        # V = ms.ops.einsum("bnhm,bnh->bnhm", V2, Z)
        V = Q.unsqueeze(3).mul(KV.unsqueeze(1)).sum(-1).mul(Z.unsqueeze(-1))
        return V

def ld(name):
        return ms.Tensor(np.load("../../Code_ALLOT/src/" + name + ".npy"), ms.float32)

class S2TAttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(S2TAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self.gcn1 = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)
        self.gcn2 = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def construct(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads
        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        gcn_keys = self.gcn1(keys.swapaxes(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        keys = self.key_projection(gcn_keys.swapaxes(-1, 1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        gcn_values = self.gcn2(values.swapaxes(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        values = self.value_projection(gcn_values.swapaxes(-1, 1)).view(B, S, N2, H, -1) # [B, S, N, H, d]
        queries = queries.swapaxes(2, 1) # [B, N, L, H, d]
        keys = keys.swapaxes(2, 1) # [B, N, S, H, d]
        values = values.swapaxes(2, 1)
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, N1, L, -1)
        
        new_values = new_values.swapaxes(2, 1) # [B, L, N1, D]
        # Project the output and return
        return self.out_projection(new_values)


class STSAttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(STSAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L*N1, H, -1) # [B, L*N, H, d]
        keys = self.key_projection(keys).view(B, S*N2, H, -1)
        values = self.value_projection(values).view(B, S*N2, H, -1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, N1, -1)
        # Project the output and return
        return self.out_projection(new_values)


class STSGAttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(STSGAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads * 2, d_model)
        self.n_heads = n_heads

        self.gcn = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def construct(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape
        H = self.n_heads

        gcn_values = self.gcn(values.swapaxes(-1, 1), adj_mats, **kwargs) # [B, D, N, T]
        gcn_values = gcn_values.swapaxes(-1, 1) # [B, T, N, D]

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L*N1, H, -1) # [B, L*N, H, d]
        keys = self.key_projection(keys).view(B, S*N2, H, -1)
        values = self.value_projection(values).view(B, S*N2, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, N1, -1)

        new_values = ms.ops.cat([new_values, gcn_values], axis=-1)

        # Project the output and return
        return self.out_projection(new_values)


class T2SAttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(T2SAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape # [B, S, N, D]
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        keys = self.key_projection(keys).view(B, S, N2, H, -1)
        values = self.value_projection(values).view(B, S, N2, H, -1)

        queries = queries.swapaxes(2,1) # [B, N, L, H, d]
        keys = keys.swapaxes(2,1) # [B, N, S, H, d]
        values = values.swapaxes(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        ).view(B, N1, L, -1)

        new_values = new_values.swapaxes(2,1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)


class T2SGAttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 n_graphs=3, order=2, use_bn=True, dropout=0.1):
        super(T2SGAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.gcn = GraphConv(d_model, d_model, n_graphs, order, use_bn, dropout)

    def construct(self, queries, keys, values, attn_mask, adj_mats, **kwargs):
        # Extract the dimensions into local variables
        B, L, N1, _ = queries.shape
        _, S, N2, _ = keys.shape # [B, S, N, D]
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, L, N1, H, -1)
        keys = self.key_projection(keys).view(B, S, N2, H, -1)
        values = self.value_projection(values).view(B, S, N2, H, -1)

        queries = queries.swapaxes(2,1) # [B, N, L, H, d]
        keys = keys.swapaxes(2,1) # [B, N, S, H, d]
        values = values.swapaxes(2,1)
        
        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        ).view(B, N1, L, -1)

        new_values = self.gcn(new_values.permute(0, 3, 1, 2), adj_mats, **kwargs) # [B, D, N, L]
        new_values = new_values.swapaxes(-1, 1) # [B, L, N1, D]
        
        # Project the output and return
        return self.out_projection(new_values)