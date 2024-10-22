import mindspore
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, XavierUniform

from model import loss


def normalize(data, mean, std):
    return (data - mean) / std


def unnormalize(data, mean, std):
    return data * std + mean


def get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = list(full_seq.shape)[1]
    # indices = LongTensor(seq_len).to(device)
    indices = mindspore.Tensor(seq_len)
    # arange(0, seq_len, out=indices)
    indices = np.arange(0, seq_len)
    indices = Parameter(indices, requires_grad=False)
    for i, row in enumerate(full_seq):
        if i == 0:
            first_seq = full_seq[0][kernel_size - 1:]
            first_seq = first_seq.expand_dims(axis=0)
            second_seq = full_seq[0][:-kernel_size + 1]
            second_seq = second_seq.expand_dims(axis=0)
        else:
            first_seq = ops.concat((first_seq, full_seq[i][kernel_size - 1:].expand_dims(axis=0)))
            second_seq = ops.concat((second_seq, full_seq[i][:-kernel_size + 1].expand_dims(axis=0)))

    local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq


class Attr(nn.Cell):
    def __init__(self, embed_dims, data_feature):
        super(Attr, self).__init__()
        self.embed_dims = embed_dims
        self.data_feature = data_feature
        for name, dim_in, dim_out in self.embed_dims:
            self.insert_child_to_cell(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for _, _, dim_out in self.embed_dims:
            sz += dim_out
        return sz + 1

    def construct(self, batch):
        em_list = []
        for name, _, _ in self.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = batch[name]
            attr_t = ops.Cast()(attr_t, mindspore.int64)
            attr_t = ops.squeeze(embed(attr_t))
            em_list.append(attr_t)
        dist_mean, dist_std = self.data_feature["dist_mean"], self.data_feature["dist_std"]
        dist = normalize(batch["dist"], dist_mean, dist_std)
        dist = normalize(dist, dist_mean, dist_std)
        em_list.append(dist)
        # # print("Attr em_list:" + str(em_list))
        return ops.concat(em_list, axis=1)


class GeoConv(nn.Cell):
    def __init__(self, kernel_size, num_filter, data_feature={}):
        super(GeoConv, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.data_feature = data_feature
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Dense(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size, pad_mode="pad")

    def construct(self, batch):
        longi_mean, longi_std = self.data_feature["longi_mean"], self.data_feature["longi_std"]
        current_longi = normalize(batch["current_longi"], longi_mean, longi_std)
        # print("GeoConv current_longi shape " + str(current_longi.shape))
        lngs = ops.expand_dims(current_longi, axis=2)
        # print("GeoConv lngs shape " + str(lngs.shape))
        lati_mean, lati_std = self.data_feature["lati_mean"], self.data_feature["lati_std"]
        current_lati = normalize(batch["current_lati"], lati_mean, lati_std)
        lats = ops.expand_dims(current_lati, axis=2)
        states = self.state_em(ops.Cast()(batch['current_state'], mindspore.int64))
        # print("GeoConv states shape " + str(states.shape))
        states = ops.Cast()(states, mindspore.float32)
        # self.weight = ops.Cast()(self.weight, mindspore.float64)
        locs = ops.concat((lngs, lats, states), axis=2)
        # print("GeoConv locs shape " + str(locs.shape))
        locs = ops.Cast()(locs, mindspore.float32)
        locs = ops.tanh(self.process_coords(locs))
        locs = locs.transpose((0, 2, 1))
        # print("GeoConv locs2 shape " + str(locs.shape))
        conv_locs = self.conv(locs)
        # print("GeoConv self.num_filter = " + str(self.num_filter))
        # print("GeoConv self.kernel_size = " + str(self.kernel_size))
        # print("GeoConv conv_locs shape " + str(conv_locs.shape))
        conv_locs = ops.Elu()(conv_locs)
        # print("GeoConv conv_locs shape " + str(conv_locs.shape))
        conv_locs = conv_locs.transpose(0, 2, 1)
        # print("GeoConv conv_locs shape " + str(conv_locs.shape))
        dist_gap_mean, dist_gap_std = self.data_feature["dist_gap_mean"], self.data_feature["dist_gap_std"]
        current_dis = normalize(batch["current_dis"], dist_gap_mean, dist_gap_std)
        # print("GeoConv current_dis shape" + str(current_dis.shape))
        local_dist = get_local_seq(current_dis, self.kernel_size, dist_gap_mean, dist_gap_std)
        # print("GeoConv local_dist's shape" + str(local_dist.shape))
        local_dist = ops.expand_dims(local_dist, axis=2)
        local_dist = ops.Cast()(local_dist, mindspore.float32)
        conv_locs = ops.concat((conv_locs, local_dist), axis=2)
        return conv_locs


class SpatioTemporal(nn.Cell):
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention',
                 rnn_type='LSTM', rnn_num_layers=1, hidden_size=128,
                 data_feature={}):
        super(SpatioTemporal, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.hidden_size = hidden_size
        self.data_feature = data_feature
        self.geo_conv = GeoConv(
            kernel_size=kernel_size,
            num_filter=num_filter,
            data_feature=data_feature
        )
        # num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=num_filter + 1 + attr_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
            )
        elif rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(
                input_size=num_filter + 1 + attr_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        else:
            raise ValueError('invalid rnn_type, please select RNN or LSTM')
        if pooling_method == 'attention':
            self.attr2atten = nn.Dense(attr_size, hidden_size)

    def out_size(self):
        return self.hidden_size

    def mean_pooling(self, hiddens, lens):
        hiddens = ops.reduce_sum(hiddens, axis=1, keep_dims=False)
        lens = mindspore.Tensor(lens)
        lens = Parameter(ops.expand_dims(lens, axis=1), requires_grad=False)
        hiddens = hiddens / lens
        return hiddens

    def atten_pooling(self, hiddens, attr_t):
        atten = ops.tanh(self.attr2atten(attr_t)).transpose((0, 2, 1))
        alpha = ops.BatchMatMul()(hiddens, atten)
        alpha = ops.exp(-alpha)
        alpha = alpha / ops.ReduceSum(keep_dims=True)(alpha, axis=1)
        hiddens = ops.transpose(hiddens, (0, 2, 1))
        hiddens = ops.BatchMatMul()(hiddens, alpha)
        hiddens = ops.squeeze(hiddens)
        return hiddens

    def construct(self, batch, attr_t):
        conv_locs = self.geo_conv(batch)

        attr_t = ops.expand_dims(attr_t, axis=1)
        expand_attr_t = attr_t.broadcast_to(conv_locs.shape[:2] + (attr_t.shape[-1],))

        # concat the loc_conv and the attributes

        conv_locs = ops.concat((conv_locs, expand_attr_t), axis=2)

        lens = [batch["current_longi"].shape[1]] * batch["current_longi"].shape[0]
        lens = list(map(lambda x: x - self.kernel_size + 1, lens))

        packed_inputs = conv_locs
        packed_hiddens, _ = self.rnn(packed_inputs)
        lens = []
        for row in packed_hiddens:
            lens.append(len(row))
        lens = mindspore.Tensor(lens)
        hiddens = packed_hiddens
        packed_hiddens = ops.reshape(packed_hiddens, (-1, 128))
        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
        else:
            return packed_hiddens, lens, self.atten_pooling(hiddens, attr_t)


class EntireEstimator(nn.Cell):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(EntireEstimator, self).__init__()
        self.input2hid = nn.Dense(input_size, hidden_size)
        self.residuals = nn.CellList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Dense(hidden_size, hidden_size))
        self.hid2out = nn.Dense(hidden_size, 1)
        self.loss_fn=nn.L1Loss()

    def construct(self, attr_t, sptm_t):
        inputs = ops.concat((attr_t, sptm_t), axis=1)
        hidden = nn.LeakyReLU()(self.input2hid(inputs))
        for i in range(len(self.residuals)):
            residual = nn.LeakyReLU()(self.residuals[i](hidden))
            hidden = hidden + residual
        out = self.hid2out(hidden)
        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label
        label = label * std + mean
        pred = pred * std + mean
        return self.loss_fn(pred, label)


class LocalEstimator(nn.Cell):
    def __init__(self, input_size, eps=10):
        super(LocalEstimator, self).__init__()
        self.input2hid = nn.Dense(input_size, 64)
        self.hid2hid = nn.Dense(64, 32)
        self.hid2out = nn.Dense(32, 1)
        self.eps = eps
        self.loss_fn=nn.L1Loss()

    def construct(self, sptm_s):
        # # print("LocalEstimator sptm_s shape " + str(sptm_s.shape))
        hidden = nn.LeakyReLU()(self.input2hid(sptm_s))
        # # print("LocalEstimator hidden1 shape " + str(hidden.shape))
        hidden = nn.LeakyReLU()(self.hid2hid(hidden))
        # # print("LocalEstimator hidden2 shape " + str(hidden.shape))
        out = self.hid2out(hidden)
        # # print("LocalEstimator out shape " + str(out.shape))
        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = ops.reshape(label, (-1,))
        label = label * std + mean
        pred = pred * std + mean
        local_loss=self.loss_fn(pred,label)
        return local_loss


class DeepTTE_model(nn.Cell):
    def __init__(self, config, data_feature):
        super(DeepTTE_model, self).__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        uid_emb_size = config.get("uid_emb_size", 16)
        weekid_emb_size = config.get("weekid_emb_size", 3)
        timdid_emb_size = config.get("timdid_emb_size", 8)
        uid_size = data_feature.get("uid_size", 24000)
        embed_dims = [
            ('uid', uid_size, uid_emb_size),
            ('weekid', 7, weekid_emb_size),
            ('timeid', 1440, timdid_emb_size),
        ]
        self.kernel_size = config.get('kernel_size', 3)
        num_filter = config.get('num_filter', 32)
        pooling_method = config.get("pooling_method", "attention")
        num_final_fcs = config.get('num_final_fcs', 4)
        final_fc_size = config.get('final_fc_size', 128)
        self.alpha = config.get('alpha', 0.3)
        rnn_type = config.get('rnn_type', 'LSTM')
        rnn_num_layers = config.get('rnn_num_layers', 1)
        hidden_size = config.get('hidden_size', 256)
        self.eps = config.get('eps', 10)
        self.attr_net = Attr(embed_dims, data_feature)
        self.spatio_temporal = SpatioTemporal(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=num_filter,
            pooling_method=pooling_method,
            rnn_type=rnn_type,
            rnn_num_layers=rnn_num_layers,
            hidden_size=hidden_size,
            data_feature=data_feature
        )
        self.entire_estimate = EntireEstimator(
            input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=num_final_fcs,
            hidden_size=final_fc_size,
        )
        self.local_estimate = LocalEstimator(
            input_size=self.spatio_temporal.out_size(),
            eps=self.eps,
        )
        self._init_weight()

    def _init_weight(self):
        for params in self.get_parameters():
            # if params.name.find('.bias') != -1:
            #     params.set_data(ops.Fill()(mindspore.float32, params.shape, 0))
            if params.name.find('.weight') != -1:
                # # print(params.data)
                params.set_data(initializer(XavierUniform(), params.shape))

    def construct(self, batch,mode):
        attr_t = self.attr_net(batch)
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(batch, attr_t)

        entire_out = self.entire_estimate(attr_t, sptm_t)
        if mode=='train':
            local_out = self.local_estimate(sptm_s)
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def predict(self, batch,mode):
        time_mean, time_std = self.data_feature["time_mean"], self.data_feature["time_std"]
        if mode=="train":
            entire_out, (local_out, local_length) = self.construct(batch,mode)
            entire_out = unnormalize(entire_out, time_mean, time_std)
            return entire_out, (local_out, local_length)
        else:
            entire_out = self.construct(batch,mode)
            entire_out = unnormalize(entire_out, time_mean, time_std)
            return entire_out


class DeepTTE(nn.Cell):
    """DeepTTE Loss"""

    def __init__(self, config, data_feature):
        super(DeepTTE, self).__init__()
        self.network = DeepTTE_model(config,data_feature)
        self.reshape = ops.Reshape()
        self.mode="train"

    def set_loss(self,loss_fn):
        pass

    def train(self):
        self.mode="train"

    def eval(self):
        self.mode="eval"

    def predict(self,current_longi, current_lati, current_tim,
                  current_dis, current_state, uid, weekid,
                  timeid, dist, time, traj_len, traj_id,
                  start_timestamp):
        uid = ops.Cast()(uid, mindspore.float32)
        weekid = ops.Cast()(weekid, mindspore.float32)
        timeid = ops.Cast()(timeid, mindspore.float32)
        current_longi = ops.Cast()(current_longi, mindspore.float32)
        current_lati = ops.Cast()(current_lati, mindspore.float32)
        current_tim = ops.Cast()(current_tim, mindspore.float32)
        current_dis = ops.Cast()(current_dis, mindspore.float32)
        current_state = ops.Cast()(current_state, mindspore.float32)
        dist = ops.Cast()(dist, mindspore.float32)
        time = ops.Cast()(time, mindspore.float32)
        traj_len = ops.Cast()(traj_len, mindspore.float32)
        traj_id = ops.Cast()(traj_id, mindspore.float32)
        start_timestamp = ops.Cast()(start_timestamp, mindspore.float32)
        batch = {'current_longi': current_longi,
                 'current_lati': current_lati,
                 'current_tim': current_tim,
                 'current_dis': current_dis,
                 'current_state': current_state,
                 'uid': uid,
                 'weekid': weekid,
                 'timeid': timeid,
                 'dist': dist,
                 'time': time,
                 'traj_len': traj_len,
                 'traj_id': traj_id,
                 'start_timestamp': start_timestamp}
        return self.network.predict(batch,self.mode)

    def construct(self, current_longi, current_lati, current_tim,
                  current_dis, current_state, uid, weekid,
                  timeid, dist, time, traj_len, traj_id,
                  start_timestamp):
        uid = ops.Cast()(uid, mindspore.float32)
        weekid = ops.Cast()(weekid, mindspore.float32)
        timeid = ops.Cast()(timeid, mindspore.float32)
        current_longi = ops.Cast()(current_longi, mindspore.float32)
        current_lati = ops.Cast()(current_lati, mindspore.float32)
        current_tim = ops.Cast()(current_tim, mindspore.float32)
        current_dis = ops.Cast()(current_dis, mindspore.float32)
        current_state = ops.Cast()(current_state, mindspore.float32)
        dist = ops.Cast()(dist, mindspore.float32)
        time = ops.Cast()(time, mindspore.float32)
        traj_len = ops.Cast()(traj_len, mindspore.float32)
        traj_id = ops.Cast()(traj_id, mindspore.float32)
        start_timestamp = ops.Cast()(start_timestamp, mindspore.float32)
        batch = {'current_longi': current_longi,
                 'current_lati': current_lati,
                 'current_tim': current_tim,
                 'current_dis': current_dis,
                 'current_state': current_state,
                 'uid': uid,
                 'weekid': weekid,
                 'timeid': timeid,
                 'dist': dist,
                 'time': time,
                 'traj_len': traj_len,
                 'traj_id': traj_id,
                 'start_timestamp': start_timestamp}
        if self.mode=="train":
            entire_out, (local_out, local_length) = self.network.predict(batch,self.mode)
        else:
            entire_out = self.network.predict(batch,self.mode)
        time_mean, time_std = self.network.data_feature['time_mean'], self.network.data_feature["time_std"]

        entire_out = normalize(entire_out, time_mean, time_std)
        time = normalize(batch["time"], time_mean, time_std)
        entire_loss = self.network.entire_estimate.eval_on_batch(entire_out, time, time_mean, time_std)

        if self.mode=="train":
            time_gap_mean, time_gap_std = self.network.data_feature["time_gap_mean"], self.network.data_feature[
                "time_gap_std"]
            mean, std = (self.network.kernel_size - 1) * time_gap_mean, (self.network.kernel_size - 1) * time_gap_std
            current_tim = normalize(batch["current_tim"], time_gap_mean, time_gap_std)
            local_label = get_local_seq(current_tim, self.network.kernel_size, mean, std)
            local_loss = self.network.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)
            return (1 - self.network.alpha) * entire_loss + self.network.alpha * local_loss
        else:
            return entire_loss
