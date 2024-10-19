from logging import getLogger
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
import mindspore.numpy as mnp
from model import loss
from model.abstract_traffic_state_model import AbstractTrafficStateModel
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, XavierNormal,XavierUniform,Normal, Uniform
import mindspore as ms
from mindspore import Tensor, ops
from scipy.sparse.linalg import eigsh

class myLayerNorm(nn.Cell):
    def __init__(self,dim,epsilon=1e-5,axis=-1):
        super().__init__()
        self.dim=dim
        self.epsilon=epsilon
        self.gamma = Parameter(initializer("ones",(dim,)),name="gamma")
        self.beta = Parameter(initializer("zeros",(dim,)), name="beta")
        self.axis=axis
    def construct(self, x):
        x_mean = ops.mean(x,axis=self.axis,keep_dims=True)
        x -= x_mean
        x_std = ops.sqrt(ops.mean(ops.square(x),axis=self.axis,keep_dims=True)+self.epsilon)
        x = x/x_std*self.gamma+self.beta
        return x


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: Tensor, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: Tensor, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]
    # 将 W 转换为 Tensor
    if isinstance(W, np.ndarray):
        W = Tensor(W, ms.float32)

    D = ops.diag(ops.sum(W, dim =1))

    L = D - W

    # 使用 scipy 进行特征值计算
    lambda_max = eigsh(L.asnumpy(), k=1, which='LM')[0].real
    lambda_max = Tensor(lambda_max, ms.float32)

    identity = ops.eye(W.shape[0], W.shape[0], ms.float32)

    return (2 * L) / lambda_max - identity

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, Tensor, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(Tensor), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    # 初始化 T_0 和 T_1
    identity = ops.eye(N, N, ms.float32)
    cheb_polynomials = [identity, L_tilde.copy()]

    for i in range(2, K):
        # 计算 T_i
        cheb_polynomials.append(2 * ops.matmul(L_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])

    return cheb_polynomials


class SScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.sqrt = ops.Sqrt()
        self.masked_fill = ops.masked_fill
        self.transpose = ops.Transpose()

    def construct(self, Q, K, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = ops.matmul(Q, self.transpose(K,(0,1,3,2))) / self.sqrt(Tensor(self.d_k,ms.float32))
        if attn_mask is not None:
            scores = self.masked_fill(scores, attn_mask, -1e9)
        return scores


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d
        #self.matmul = ops.MatMul()
        self.sqrt = ops.Sqrt()
        self.softmax = ops.Softmax(axis=3)
        self.masked_fill = ops.masked_fill
        self.dtype = mstype.float32
        self.transpose = ops.Transpose()

    def construct(self, Q, K, V, attn_mask, res_att):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = ops.matmul(Q, self.transpose(K, (0,1,2,4,3))) / self.sqrt(Tensor(self.d_k, ms.float32)) + res_att

        if attn_mask is not None :
            scores = self.masked_fill(scores, attn_mask, -1e9)
        softmax=ops.Softmax(axis=3)
        attn = softmax(scores)
        context = ops.matmul(attn, V)
        return context, scores



class SMultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Dense(d_model, d_k * n_heads, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.W_K = nn.Dense(d_model, d_k * n_heads, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.transpose = ops.Transpose()

    def construct(self, input_Q, input_K, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.shape[0]
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q)
        Q = Q.reshape((batch_size, -1, self.n_heads, self.d_k))
        Q = self.transpose(Q, (0, 2, 1, 3)) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K)
        K = K.reshape((batch_size, -1, self.n_heads, self.d_k))
        K = self.transpose(K, (0, 2, 1, 3))# K: [batch_size, n_heads, len_k, d_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads, 1,1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn


class MultiHeadAttention(nn.Cell):
    def __init__(self,d_model, d_k, d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.W_Q = nn.Dense(d_model, d_k * n_heads, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.W_K = nn.Dense(d_model, d_k * n_heads, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.W_V = nn.Dense(d_model, d_v * n_heads, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.fc = nn.Dense(n_heads * d_v, d_model, has_bias=False,weight_init=XavierUniform(),
                                    bias_init=Uniform())
        self.transpose = ops.Transpose()
        self.LayerNorm = nn.LayerNorm(normalized_shape=(d_model,), epsilon= 1e-5)

       # self.LayerNorm = myLayerNorm(d_model)
    def construct(self, input_Q, input_K, input_V, attn_mask, res_att):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.shape[0]

        Q = self.W_Q(input_Q).view((batch_size, self.num_of_d, -1, self.n_heads, self.d_k))
        Q = self.transpose(Q, (0, 1, 3, 2, 4))
        K = self.W_K(input_K).view((batch_size, self.num_of_d, -1, self.n_heads, self.d_k))
        K = self.transpose(K, (0, 1, 3, 2, 4)) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view((batch_size, self.num_of_d, -1, self.n_heads, self.d_v))
        V = self.transpose(V, (0, 1, 3, 2, 4)) # V: [batch_size, n_heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)
        context = self.transpose(context,(0,1,3,2,4))
        context = context.reshape((batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v) ) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.LayerNorm(output + residual)
        return output, res_attn



class cheb_conv_withSAt(nn.Cell):

    '''
    K-order Chebyshev graph convolution with spatial attention
    实现了一个带有空间注意力机制的K阶Chebyshev图卷积层，
    它利用切比雪夫多项式来近似图卷积操作，
    并将空间注意力机制融入到图卷积中，以增强模型对空间依赖关系的捕捉能力。
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_of_vertices = num_of_vertices
        self.relu = nn.ReLU()
        self.transpose = ops.Transpose()
        self.Theta = ParameterTuple(
            [Parameter(initializer(XavierUniform(),shape=[in_channels, out_channels], dtype=ms.float32),
                    requires_grad=True,name=f'param_{i}')
             for i in range(K)]
        )

        self.mask = ParameterTuple(
            [Parameter(initializer(XavierUniform(),shape=[num_of_vertices, num_of_vertices], dtype=ms.float32),
                     requires_grad=True,name=f'param_{i}')
             for i in range(K)]
        )

    def construct(self, x, spatial_attention, adj_pa):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = ops.zeros((batch_size, num_of_vertices, self.out_channels), dtype=ms.float32)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:, k, :, :] +  ops.Mul()(adj_pa, mask)
                                      #adj_pa * mask
                myspatial_attention = nn.Softmax(axis=1)(myspatial_attention)

                T_k_with_at = ops.Mul()(T_k , myspatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                T_k_with_at = self.transpose(T_k_with_at,(0, 2, 1))

                rhs = ops.matmul(T_k_with_at,graph_signal)

                rhs = ops.matmul(rhs,theta_k)

                output = output + rhs

            outputs.append(output.unsqueeze(-1))
        outputs=list(outputs)
        output=ops.Concat(axis=-1)(outputs)
        output= self.relu(output)
        return output# (b, N, F_out, T)

class cheb_conv(nn.Cell):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = ParameterTuple(
            [Parameter(initializer(XavierUniform(),shape=[in_channels, out_channels], dtype=ms.float32),
                      requires_grad=True) for _ in range(K)]
        )
        self.relu = nn.ReLU()
        self.transpose = ops.Transpose()

    def construct(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = ops.zeros((batch_size, num_of_vertices, self.out_channels), dtype=ms.float32)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                graph_signal=self.transpose(graph_signal,(0, 2, 1))
                result = ops.matmul(graph_signal, T_k)
                rhs =self.transpose(result,(0, 2, 1))
                output = output + ops.matmul(rhs,theta_k)
            outputs.append(output.unsqueeze(-1))
        output=ops.Concat(axis=-1)(outputs)
        output=self.relu(output)
        return output


class Embedding(nn.Cell):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.d_Em=d_Em
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        #self.pos_embed.embedding_table.requires_grad = False
        self.norm = nn.LayerNorm((d_Em,),epsilon= 1e-5)
        self.transpose = ops.Transpose()

    def construct(self, x, batch_size):
        if self.Etype == 'T':
            pos = ops.arange(self.nb_seq, dtype=mstype.int32)
            pos = ops.unsqueeze(pos, 0)
            pos = ops.unsqueeze(pos, 0)
            pos = ops.broadcast_to(pos,(batch_size,self.num_of_features, self.nb_seq))
            x=self.transpose(x, (0, 2, 3, 1))
            #检查权重是否包含NaN
            # embedding_table_np = self.pos_embed.embedding_table.data.asnumpy()
            # if np.isnan(embedding_table_np).any():
            #     print("NaN values found in the embedding weights")
            # else:
            #     print("No NaN values in the embedding weights")
            embedding = x + self.pos_embed(pos)
        else:
            pos = ops.arange(self.nb_seq, dtype=mstype.int64)
            pos = ops.unsqueeze(pos, 0)
            pos = ops.broadcast_to(pos,(batch_size, self.nb_seq))
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx

class GTU(nn.Cell):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = ops.Tanh()
        self.sigmoid = ops.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels,
                                 kernel_size=(1, kernel_size),
                                 stride=(1, time_strides),
                                 has_bias=True,  # 包含偏置项
                                 pad_mode='valid',  # 不进行填充
                                 padding=0,
                                 weight_init=XavierUniform(),
                                 bias_init=Uniform()
                                 )

    def construct(self, x):

        x_causal_conv = self.con2out(x)

        x_p = x_causal_conv[:, :self.in_channels, :, :]

        x_q = x_causal_conv[:, -self.in_channels:, :, :]

        x_p = self.tanh(x_p)
        x_q = self.sigmoid(x_q)
        x_gtu = ops.mul(x_p, x_q)

        return x_gtu



class DSTAGNN_block(nn.Cell):
    def __init__(self, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(DSTAGNN_block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.adj_pa = Tensor(adj_pa, ms.float32)

        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d),
                                  has_bias=True,
                                  pad_mode='valid',  # 不进行填充
                                  padding=0,
                                  weight_init=XavierUniform(),
                                  bias_init=Uniform()
                                  )

        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T')
        self.EmbedS = Embedding(num_of_vertices, d_model, num_of_d, 'S')
        self.TAt = MultiHeadAttention(num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.SAt = SMultiHeadAttention( d_model, d_k, d_v, K)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)

        self.gtu3 = GTU(nb_time_filter, time_strides, 3)
        self.gtu5 = GTU(nb_time_filter, time_strides, 5)
        self.gtu7 = GTU(nb_time_filter, time_strides, 7)

        self.pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides),
                                       has_bias=True,
                                       pad_mode='valid',  # 不进行填充
                                       padding=0,
                                       weight_init=XavierUniform(),
                                       bias_init=Uniform()
                                       )

        self.dropout = nn.Dropout(p=0.05)
        self.fcmy = nn.SequentialCell(
            nn.Dense(3 * num_of_timesteps - 12, num_of_timesteps,weight_init=XavierUniform(),
                                    bias_init=Uniform()),
            nn.Dropout(p=0.05),
        )
        self.nb_time_filter=nb_time_filter
        self.ln = nn.LayerNorm(normalized_shape=(nb_time_filter,), epsilon = 1e-5)

    def construct(self, x, res_att):
        '''
        :param x: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        transpose_op = ops.Transpose()

        # TAT 时序的编码
        if num_of_features == 1:
            TEmx = self.EmbedT(x, batch_size)  # B,F,T,N
        else:
            TEmx =transpose_op(x, (0, 2,3,1))

        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)  # B,F,T,N; B,F,Ht,T,T

        #num_of_timesteps->d_model
        x_TAt = self.pre_conv(transpose_op(TATout,((0, 2, 3, 1))))[:, :, :, -1]
        x_TAt = transpose_op(x_TAt,(0, 2, 1))

        # SMultiHeadAttention
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # B,N,d_model
        SEmx_TAt = self.dropout(SEmx_TAt)  # B,N,d_model
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)  # B,Hs,N,N

        # graph convolution in spatial dim
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_pa)  # B,N,F,T

        # convolution along the time axis
        X = transpose_op(spatial_gcn,(0, 2, 1, 3))

        x_gtu = []
        gtu3 = self.gtu3(X)
        x_gtu.append(gtu3)
        gtu5 = self.gtu5(X)
        x_gtu.append(gtu5)
        gtu7 = self.gtu7(X)
        x_gtu.append(gtu7)

        time_conv = ops.Concat(axis=-1)(x_gtu)

        time_conv = self.fcmy(time_conv)

        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)  # B,F,N,T

        # residual shortcut
        if num_of_features == 1:
            x = transpose_op(x,(0, 2, 1, 3))
            x_residual = self.residual_conv(x)
        else:
            x_residual = transpose_op(x,(0, 2, 1, 3))

        x_residual = self.relu(x_residual + time_conv_output)
        x_residual = transpose_op(x_residual,(0, 3, 2, 1))
        x_residual = self.ln(x_residual)
        x_residual = transpose_op(x_residual, (0, 2, 3, 1))
        return x_residual, re_At


class DSTAGNN_model(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self._logger = getLogger()
        graph_use = config.get('graph_use', "AG")

        # data feature
        self._scaler = self.data_feature.get('scaler')
        adj_TMD = self.data_feature.get('adj_TMD')
        adj_merge = self.data_feature.get('adj_mx') if graph_use == 'G' else adj_TMD
        adj_pa = self.data_feature.get('adj_pa')
        num_of_vertices = self.data_feature.get('num_nodes')

        # model
        num_of_d = config.get('in_channels', 1)
        nb_block = config.get('nb_block', 4)
        in_channels = config.get('in_channels', 1)
        K = config.get('K', 3)
        nb_chev_filter = config.get('nb_chev_filter', 32)
        nb_time_filter = config.get('nb_time_filter', 32)
        time_strides = 1
        num_for_predict = config.get('output_window', 12)
        len_input = config.get('input_window', 12)
        d_model = config.get('d_model', 512)
        d_k = config.get('d_k', 32)
        d_v = config.get('d_k', 32)
        n_heads = config.get('n_heads', 3)

        # cheb_polynomials
        L_tilde = scaled_Laplacian(adj_merge)

        cheb_polynomials = [Tensor(i, ms.float32) for i in cheb_polynomial(L_tilde, K)]

        self.BlockList = nn.CellList([DSTAGNN_block( num_of_d, in_channels, K,
                                                      nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                                                      adj_pa, adj_TMD, num_of_vertices, len_input, d_model, d_k, d_v,
                                                      n_heads)])

        self.BlockList.extend([DSTAGNN_block( num_of_d * nb_time_filter, nb_chev_filter, K,
                                             nb_chev_filter, nb_time_filter, 1, cheb_polynomials,
                                             adj_pa, adj_TMD, num_of_vertices, len_input // time_strides, d_model, d_k,
                                             d_v, n_heads) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int((len_input / time_strides) * nb_block), 128,
                                    kernel_size=(1, nb_time_filter),pad_mode='valid',  # 不进行填充
                                    has_bias=True,
                                    padding=0,
                                    weight_init=XavierUniform(),
                                    bias_init=Uniform()
                                    )
        self.final_fc = nn.Dense(128, num_for_predict,
                                weight_init=XavierUniform(),
                                bias_init=Uniform() )


    def construct(self, x):
        """

        @param x: (B, N_nodes, F_in, T_in)
        @return: (B, N_nodes, T_out)
        """

        need_concat = []
        res_att = 0
        transpose_op = ops.Transpose()

        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)

        final_x = ops.Concat(axis=-1)(need_concat)

        final_x = transpose_op(final_x,(0, 3, 1, 2))

        output = self.final_conv(final_x)

        output = output[:, :, :, -1]

        output = transpose_op(output,(0, 2, 1))

        output = self.final_fc(output)

        return output

class DSTAGNN(nn.Cell):
    def __init__(self, config, data_feature):
        super(DSTAGNN, self).__init__()
        self.loss = loss.smooth_l1_loss
        self.network = DSTAGNN_model(config, data_feature)
        self.reshape = ops.Reshape()
        self.mode = "train"
        self.zscore = data_feature['scaler']
        self.transpose = ops.Transpose()


    def train(self):
        self.mode = "train"
        self.set_grad(True)
        self.set_train(True)

    def eval(self):
        self.mode = "eval"
        self.set_grad(False)
        self.set_train(False)

    def validate(self):
        self.set_grad(False)
        self.set_train(False)

    def predict(self,x,label):
        x = self.transpose(x,(0, 2, 3, 1))
        x=x[...,0:1,:]
        label=label[...,0:1,:]
        y_predict= self.network(x)
        y_predict = self.transpose(y_predict,(0, 2, 1)).unsqueeze(-1)
        y_predict = self.zscore.inverse_transform(y_predict)
        label = self.zscore.inverse_transform(label)
        return y_predict[...,0:1],label

    def calculate_loss(self,x,label):
        assert not mnp.isnan(x).any(), "Input data contains NaN values"
        #print("xxx",x.shape)
        x=x[...,0:1]
        label=label[...,0:1]
        x = self.transpose(x, (0, 2, 3, 1))
        y = self.network(x)
        y = self.transpose(y, (0, 2, 1)).unsqueeze(-1)
        #print(y.shape,label.shape)
        y = self.zscore.inverse_transform(y)
        label = self.zscore.inverse_transform(label)
        loss = self.loss(y, label,0)
        return loss

    def construct(self, x, label):
        if self.mode=="train":
            return self.calculate_loss(x,label)
        elif self.mode=="eval":
            return self.predict(x,label)


