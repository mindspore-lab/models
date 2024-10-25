import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, Parameter
import numpy

class LSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, has_bias=True, batch_first=True, dropout=0):
        super(LSTM, self).__init__()

        if not 0 <= dropout <= 1:
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_op = nn.Dropout(float(1 - dropout))
        self.has_bias = has_bias

        stdv = 1 / self.hidden_size

        self.Wi = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, input_size + hidden_size)).astype(numpy.float32)), name='Wi')
        self.Wo = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, input_size + hidden_size)).astype(numpy.float32)), name='Wo')
        self.Wf = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, input_size + hidden_size)).astype(numpy.float32)), name='Wf')
        self.Wc = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, input_size + hidden_size)).astype(numpy.float32)), name='Wc')
        self.bi = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, 1)).astype(numpy.float32)), name='bi')
        self.bo = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, 1)).astype(numpy.float32)), name='bo')
        self.bf = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, 1)).astype(numpy.float32)), name='bf')
        self.bc = Parameter(Tensor(numpy.random.uniform(-stdv, stdv, (hidden_size, 1)).astype(numpy.float32)), name='bc')

        self.concat_0 = ops.Concat(axis=0)
        self.concat_1 = ops.Concat(axis=1)
        self.squeeze_0 = ops.Squeeze(axis=0)
        self.transpose = ops.Transpose()
        self.tensor_order = (1, 0, 2)
        self.matrix_order = (1, 0)
        self.reshape = ops.Reshape()
        self.sigmoid = ops.Sigmoid()
        self.tanh = ops.Tanh()
        self.select=ops.Select()
        self.cast=ops.Cast()
        self.Broadcastto=ops.BroadcastTo((self.hidden_size, -1))
        self.zerolike=ops.ZerosLike()

        self.matmul = ops.MatMul(False, True)  # !

    def lstm_cell(self, input_x, hidden):
        """
        input: [batch_size, input_size]
        hidden: tuple([batch_size, hidden_size], [batch_size, hidden_size])
        w_ih : [4 * hidden_size, hidden_size + input_size]
        b_ih : [4 * hidden_size]
        """
        # print(input_x.dtype)
        hx, cx = hidden
        batch_size = hx.shape[0]
        cx = self.reshape(cx, (self.hidden_size, batch_size))

        inputs = self.concat_1((hx, input_x))
        f_t = self.sigmoid(self.matmul(self.Wf, inputs) + self.bf)
        i_t = self.sigmoid(self.matmul(self.Wi, inputs) + self.bi)
        o_t = self.sigmoid(self.matmul(self.Wo, inputs) + self.bo)
        cy = f_t * cx + i_t * self.tanh(self.matmul(self.Wc, inputs) + self.bc)
        hy = o_t * self.tanh(cy)

        hy = self.reshape(hy, (batch_size, self.hidden_size))
        cy = self.reshape(cy, (batch_size, self.hidden_size))

        return hy, cy

    def construct(self, input_x, hidden, seq_length):
        """
        inputx : [seq_length, batch_size, input_size]
        hidden : ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
        seq_length : [batch_size,]
        """
        if self.batch_first:
            input_x = self.transpose(input_x, self.tensor_order)
        h, c = hidden[0][0], hidden[1][0]
        time_step = input_x.shape[0]

        seq_length = self.cast(seq_length, ms.int32)
        seq_length = self.Broadcastto(seq_length)
        seq_length = self.transpose(seq_length, (1, 0))
        
        zero_output = self.zerolike(h)
        outputs = ops.Zeros()((time_step, h.shape[0], h.shape[1]), input_x.dtype)
        state_t = (h, c)
        t = ops.ScalarToTensor()(0, ms.int64)

        while t < time_step:

            x_t = input_x[t]
            h_t = self.lstm_cell(x_t, state_t)

            seq_cond = seq_length > t
            state_t_0 = self.select(seq_cond, h_t[0], state_t[0])
            state_t_1 = self.select(seq_cond, h_t[1], state_t[1])
            output = self.select(seq_cond, h_t[0], zero_output)
            state_t = (state_t_0, state_t_1)
            outputs[t]=output
            t += 1

        if self.batch_first:
            outputs = self.transpose(outputs, self.tensor_order)
        return outputs, state_t

class Attn(nn.Cell):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Dense(self.hidden_size, self.hidden_size)
        else:  # self.method == 'concat':
            self.attn = nn.Dense(self.hidden_size * 2, self.hidden_size)
            self.other = ms.Parameter(ms.Tensor(self.hidden_size, ms.float32))  
        
        self.bmm=ops.BatchMatMul()

    def construct(self, out_state, history):
        # out_state (20, 1831, 500)
        # history (20, 49, 500)
        """[summary]
        Args:
            out_state (tensor): batch_size * state_len * hidden_size
            history (tensor): batch_size * history_len * hiddden_size
        Returns:
            [tensor]: (batch_size, state_len, history_len)
        """
        if self.method == 'dot':
            history = ops.transpose(history, (0, 2, 1))  # batch_size * hidden_size * history_len (20, 500, 49)
            attn_energies = self.bmm(out_state, history)  # (20, 1831, 49)
        else:  # self.method == 'general':
            history = self.attn(history)  # (20, 49, 500)
            history = ops.transpose(history, (0, 2, 1))  # (20, 500, 49)
            bmm = ops.BatchMatMul()
            attn_energies = bmm(out_state, history)  # (20, 1831, 49)
        return ops.Softmax(axis=2)(attn_energies)


class DeepMove_model(nn.Cell):
    """rnn model with long-term history attention"""

    def __init__(self, config, data_feature):
        super(DeepMove_model, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.uid_size = data_feature['uid_size']
        self.uid_emb_size = config['uid_emb_size']
        self.hidden_size = config['hidden_size']
        self.attn_type = config['attn_type']
        self.rnn_type = config['rnn_type']
        self.evaluate_method = config['evaluate_method']
        self.batch_size = config['batch_size']

        self.emb_loc = nn.Embedding(
            self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(
            self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(
            self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Dense(input_size, self.hidden_size)
        self.softmax_out = nn.LogSoftmax(axis=1)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1,batch_first=True)  # fixme 这里将batch_first设置成True，这样可以省去你后文中的维度变换（transpose），没有检验过是否正确。
            self.rnn_decoder = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1,batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size,1,batch_first=True)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size,1,batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Dense(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(keep_prob=1 - config['dropout_p'])

    def construct(self, history_loc, history_tim, current_loc, current_tim, loc_len, history_len, uid):

        

        # current_loc (20, 49) current_tim (20, 49) (batch_size, loc_size)
 
        loc_emb = self.emb_loc(current_loc)  # (20, 49, 500)
        tim_emb = self.emb_tim(current_tim)  # (20, 49, 40)
        # change batch * seq * input_size to seq * batch * input_size
        x = ops.Concat(2)((loc_emb, tim_emb))  # (20,9,540)
        x = self.dropout(x)  # (20,9,540)

        history_loc_emb = self.emb_loc(history_loc)  # (20, 1831, 500) 
        history_tim_emb = self.emb_tim(history_tim)  # (20, 1831, 40)
        history_x = ops.Concat(2)((history_loc_emb, history_tim_emb))
        history_x = self.dropout(history_x)

        h1 = np.zeros((1, self.batch_size, self.hidden_size),ms.float32)
        h2 = np.zeros((1, self.batch_size, self.hidden_size),ms.float32)
        c1 = np.zeros((1, self.batch_size, self.hidden_size),ms.float32)
        c2 = np.zeros((1, self.batch_size, self.hidden_size),ms.float32)


        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x)  # [batch,window,hidden][2,9,5]
            hidden_state, h2 = self.rnn_decoder(history_x)  # [batch,window,hidden][2,9,5]
        else:  # self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(history_x, (h1, c1), history_len)
            hidden_state, (h2, c2) = self.rnn_decoder(x, (h2, c2), loc_len)

        attn_weights = self.attn(hidden_state, hidden_history)  # (20, 1831, 49)
        # batch_size * state_len * input_size
        bmm = ops.BatchMatMul()
        context = bmm(attn_weights, hidden_history)  # (20, 1831, 500)
        # batch_size * state_len * 2 x input_size
        out = ops.Concat(2)((hidden_state, context))  # (20, 1831, 1000)

        origin_len = loc_len
        final_out_index = ms.Tensor(origin_len) - 1  # (20, )
        final_out_index = ops.reshape(final_out_index, (final_out_index.shape[0], 1, -1))  # (20, 1, 1)
        final_out_index = np.tile(final_out_index, (1, 1, 2 * self.hidden_size))  # (20, 1, 1000)
        out = np.squeeze(ops.gather_d(out, 1, final_out_index), axis=1)  # batch_size * (2*hidden_size) (20, 1000)
        uid_emb = self.emb_uid(uid)  # (20, uid_emb_size)
        out = ops.Concat(1)((out, uid_emb))  # (20, 1000+uid_emb_size)

        out = self.dropout(out)
        y = self.fc_final(out)  # batch_size * loc_size (20, 1831)
        score = self.softmax_out(y)
        return score

    def predict(self, history_loc, history_tim, loc, tim, target, target_tim, uid, loc_len, history_len):
        score = self.construct(history_loc, history_tim, loc, tim, loc_len, history_len, uid)
        return score

    def predict5th(self, history_loc, history_tim, loc, tim, loc_len, history_len, uid):
        score = self.construct(history_loc, history_tim, loc, tim, loc_len, history_len, uid)
        _, pre = ops.TopK()(score, 5)
        return pre


class DeepMove(nn.Cell):

    def __init__(self, config,data_feature):
        super().__init__()
        self.network = DeepMove_model(config,data_feature)
        # ms.amp.auto_mixed_precision(self.network, amp_level='O2')
        self.loc_size = self.network.loc_size
        self.weight = ops.cast(np.ones(self.loc_size) / self.loc_size, ms.float32)
        self.batch_size = self.network.batch_size
        self.mode="train"

    def set_loss(self,loss_fn):
        pass

    def train(self):
        self.mode="train"

    def eval(self):
        self.mode="eval"

    def calculate_loss(self, history_loc, history_tim, loc, tim, target, target_tim, uid, loc_len, history_len):
        criterion = ops.NLLLoss()
        scores = self.network(history_loc, history_tim, loc, tim, loc_len, history_len, uid)
        loss, _ = criterion(ops.cast(scores, ms.float32), ops.cast(target, ms.int32),
                            self.weight)  # scores.shape=[batch,loc_size] target.shape=[4470]
        return loss

    def construct(self, history_loc, history_tim, loc, tim, target, target_tim, uid, loc_len, history_len):
        t=ops.max(history_len)[0]
        history_loc=history_loc[:,:t+1]
        history_tim=history_tim[:,:t+1]
        if self.mode=="train":
            return self.calculate_loss(history_loc, history_tim, loc, tim, target, target_tim, uid, loc_len, history_len)
        elif self.mode=="eval":
            return self.network.predict(history_loc, history_tim, loc, tim, target, target_tim, uid, loc_len, history_len)