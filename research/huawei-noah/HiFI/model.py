import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform
import numpy as np


class HIFIC(nn.Cell):
    def __init__(self, params):
        super(HIFIC, self).__init__(auto_prefix=True)
        self.params=params
        self.seq_len = params['seq_len']
        self.user_hist_len = params['user_hist_len']
        self.item_nums = params['item_nums']
        self.channel_nums = params['channel_nums']
        self.dense_dim = params['dense_dim']
        self.user_nums = params['user_nums']
        self.keep_prob = params['keep_prob']
        self.gamma = params['gamma']
        self.use_rcpo = params['use_rcpo']
        self.crr_type = params['crr_type']
        self.crr_ratio_upper_bound = 20.
        self.critic_weight = params['critic_weight']

        self.emb_dim = params['emb_dim']
        self.mlp_dim = params['mlp_dim']

        self.use_gal = params['use_gal']
        self.num_heads = params['num_heads']

        # id embeddings
        self.item_id_emb = nn.Embedding(vocab_size=self.item_nums, embedding_size=self.emb_dim)
        self.channel_id_emb = nn.Embedding(vocab_size=self.channel_nums, embedding_size=self.emb_dim)
        self.pos_emb = nn.Embedding(vocab_size=self.seq_len, embedding_size=self.emb_dim)

        # networks
        ## states
        self.hist_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim*2+self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.user_hist_len,keep_prob=self.keep_prob)
        self.cand_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim * 2 + self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.channel_nums*self.seq_len,keep_prob=self.keep_prob)
        self.rank_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim * 2 + self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.seq_len,keep_prob=self.keep_prob)

        self.actor_predict = Actor(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)
        self.critic_predict = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)
        self.actor_target = Actor(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)
        self.critic_target = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)

        for params in self.actor_target.trainable_params():
            params.requires_grad=False
        for params in self.critic_target.trainable_params():
            params.requires_grad=False

        if self.use_rcpo:
            self.cost_predict = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)
            self.cost_target = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim+1, out_dims=self.channel_nums, keep_prob=self.keep_prob)
            for params in self.cost_target.trainable_params():
                params.requires_grad = False
            self.rcpo_lambda = ms.Parameter(ms.Tensor(0., ms.float32), name="rcpo_lambda")

        if self.use_gal:
            self.pre_prj = nn.Dense(self.emb_dim*2+self.dense_dim, self.mlp_dim)
            self.state_prj = nn.Dense(self.mlp_dim*3+self.emb_dim+1, self.mlp_dim)
            self.gal = GatedAttention(mlp_dim=self.mlp_dim, emb_dim=self.emb_dim, num_heads=self.num_heads, keep_prob=self.keep_prob, seq_len=self.seq_len)

    def construct(self, all_states, actions, rewards, dones):
        actions = actions.reshape((-1,1)).astype("int32")
        rewards = rewards.reshape((-1,1)).astype("float32")
        dones = dones.reshape((-1,1))

        # build current state and next state
        cur_hist = all_states[0].reshape((-1,self.user_hist_len,2+self.dense_dim))
        cur_cand, next_cand = all_states[1][:,:-1].reshape((-1,self.channel_nums, self.seq_len, 2+self.dense_dim)), \
            all_states[1][:,1:].reshape((-1,self.channel_nums, self.seq_len, 2+self.dense_dim))
        cur_ranked, next_ranked = all_states[2][:,:-1].reshape((-1,self.seq_len, 2+self.dense_dim)), \
            all_states[2][:,1:].reshape((-1,self.seq_len, 2+self.dense_dim))
        cur_pos, next_pos = all_states[3][:,:-1].reshape((-1,1)), all_states[3][:,1:].reshape((-1,1))
        cur_js, next_js = all_states[4][:,:-1].reshape((-1,1)), all_states[4][:,1:].reshape((-1,1))
        cur_ch, next_ch = all_states[5][:,:-1].reshape((-1,1)), all_states[5][:,1:].reshape((-1,1))

        merged_cur_state, cand_state, hist_state = self.build_data_embeddings(cur_hist, cur_cand, cur_ranked, cur_pos, cur_js)
        merged_next_state, next_cand_state, next_hist_state = self.build_data_embeddings(cur_hist, next_cand, next_ranked, next_pos, next_js)

        # =====learn critic=====
        q_value = self.critic_predict(merged_cur_state)
        if self.use_gal:
            next_cand_state = self.pre_prj(next_cand_state)
            tmp_next_state = self.state_prj(merged_next_state)
            gal_res = self.gal(tmp_next_state, next_hist_state, next_cand_state[:,0], next_cand_state[:,1])
            target_pi = ops.stop_gradient(self.actor_target(gal_res))
        else:
            target_pi = ops.stop_gradient(self.actor_target(merged_next_state))
        target_q_value = ops.stop_gradient(self.critic_target(merged_next_state))

        # compute critic loss
        q_a_value = q_value[ops.arange(q_value.shape[0]), actions.flatten()]
        expected_next_q = ops.sum(target_q_value * target_pi, dim=-1)
        # print("q_a_value:", q_a_value.shape)
        # print("expected next q:", expected_next_q.shape)
        # This is expected sarsa
        diff = q_a_value - (rewards.flatten() + self.gamma * (
                1. - dones.flatten()) * expected_next_q)

        if self.use_rcpo:
            cost_value = self.cost_predict(merged_cur_state)
            target_cost_value = self.cost_target(merged_next_state)
            cost_a_value = cost_value[ops.arange(cost_value.shape[0]), actions.flatten()]
            expected_next_cost = ops.sum(ops.stop_gradient(target_cost_value) * target_pi, dim=-1)
            cur_cost = (cur_js-0.02).abs()  # diff between video cur and target occupancy
            cost_diff = cost_a_value - (cur_cost + self.gamma * (
            1. - dones.flatten()) * expected_next_cost)

        critic_loss = ops.mean(diff.square())
        if self.use_rcpo:
            critic_loss+=ops.mean(cost_diff.square())

        # =====learn actor=====
        if self.use_gal:
            cand_state = self.pre_prj(cand_state)
            tmp_cur_state = self.state_prj(merged_cur_state)
            gal_res_cur = self.gal(tmp_cur_state, hist_state, cand_state[:,0], cand_state[:,1])
            pi = self.actor_predict(gal_res_cur)
        else:
            pi = self.actor_predict(merged_cur_state)

        if self.use_rcpo:
            adv = (q_a_value-self.rcpo_lambda*cost_a_value) - ops.sum(pi * (q_value-self.rcpo_lambda*cost_value), dim=-1)
        else:
            adv = q_a_value - ops.sum(pi * q_value, dim=-1)

        # This is what CRR proposed
        if self.crr_type == 'exp':
            policy_loss_coef_t = ops.Minimum()(
                (adv / self.crr_beta).exp(), self.crr_ratio_upper_bound)
        elif self.crr_type == 'binary':
            policy_loss_coef_t = (adv > 0).astype(q_a_value.dtype)
        else:
            policy_loss_coef_t = 1.

        policy_loss_coef_t = ops.stop_gradient(policy_loss_coef_t)
        # print("pi", pi.shape)
        policy_loss = -ops.log(pi[ops.arange(pi.shape[0]), actions.flatten()].squeeze() + 1.e-8)
        actor_loss = ops.mean(policy_loss * policy_loss_coef_t)

        pred = ops.argmax(pi, dim=-1)

        loss = actor_loss + self.critic_weight * critic_loss

        if self.use_rcpo:
            rcpo_grad = ops.stop_gradient(ops.mean(cost_a_value - 0.02))
        else:
            rcpo_grad = 0.
        return loss, rcpo_grad, pred

    def build_data_embeddings(self, hist, cand, ranked, pos, js, is_test=False):
        hist_id_emb = self.item_id_emb(hist[:,:,0].astype("int32"))
        hist_ch_emb = self.channel_id_emb(hist[:,:,1].astype("int32"))
        cand_id_emb = self.item_id_emb(cand[:,:,:,0].astype("int32"))
        cand_ch_emb = self.channel_id_emb(cand[:,:,:,1].astype("int32"))
        ranked_id_emb = self.item_id_emb(ranked[:,:,0].astype("int32"))
        ranked_ch_emb = self.channel_id_emb(ranked[:,:,1].astype("int32"))
        pos_emb = self.pos_emb(pos.astype("int32"))

        hist_emb = ops.cat([hist_id_emb, hist_ch_emb, hist[:,:,2:].astype("float32")], axis=-1)
        cand_emb = ops.cat([cand_id_emb, cand_ch_emb, cand[:,:,:,2:].astype("float32")], axis=-1)
        ranked_emb = ops.cat([ranked_id_emb, ranked_ch_emb, ranked[:,:,2:].astype("float32")], axis=-1)

        merged_state, cand_state, hist_state = self.build_input_states(hist_emb, cand_emb, ranked_emb, pos_emb, js, is_test)

        return merged_state, cand_state, hist_state

    def build_input_states(self, hist_emb, cand_emb, ranked_emb, pos_emb, js_feat, is_test=False):
        hist_state = self.hist_mlp(hist_emb)
        cand_state = self.cand_mlp(cand_emb.reshape((-1,self.channel_nums*self.seq_len,self.emb_dim*2+self.dense_dim)))
        ranked_state = self.rank_mlp(ranked_emb)
        if is_test:
            hist_state = hist_state.reshape((-1, self.mlp_dim))
        else:
            hist_state = hist_state.unsqueeze(1).repeat(self.seq_len-1,axis=1).reshape((-1,self.mlp_dim))
        # print(hist_state.shape, cand_state.shape, ranked_state.shape, pos_emb.shape, js_feat.shape)
        merged_state = ops.cat(
            [hist_state, cand_state, ranked_state, pos_emb.reshape((-1,self.emb_dim)), js_feat.astype("float32")], axis=-1)
        return merged_state, cand_emb.reshape((-1,self.channel_nums,self.seq_len,self.emb_dim*2+self.dense_dim)), hist_state

    def infer(self, all_states):
        # build current state and next state
        cur_hist = all_states[0].reshape((-1,self.user_hist_len,2+self.dense_dim))
        cur_cand = all_states[1].reshape((-1,self.channel_nums, self.seq_len, 2+self.dense_dim))
        cur_ranked = all_states[2].reshape((-1,self.seq_len, 2+self.dense_dim))
        cur_pos = all_states[3].reshape((-1,1))
        cur_js = all_states[4].reshape((-1,1))

        merged_cur_state, cand_state, hist_state  = self.build_data_embeddings(cur_hist, cur_cand, cur_ranked, cur_pos, cur_js, is_test=True)

        # =====learn actor=====
        if self.use_gal:
            cand_state = self.pre_prj(cand_state)
            tmp_cur_state = self.state_prj(merged_cur_state)
            gal_res_cur = self.gal(tmp_cur_state, hist_state, cand_state[:,0], cand_state[:,1])
            pi = self.actor_predict(gal_res_cur)
        else:
            pi = self.actor_predict(merged_cur_state)

        pred = ops.argmax(pi, dim=-1)

        return pred, pi


class StateMLP(nn.Cell):
    def __init__(self, mlp_dim, in_dims, out_dims, reshaped_dim, keep_prob):
        super(StateMLP,self).__init__(auto_prefix=True)
        self.bn1=nn.BatchNorm1d(in_dims)
        self.dense1 = nn.Dense(in_dims, mlp_dim)
        self.ac1 = nn.ReLU()
        self.dp1 = nn.Dropout(keep_prob=keep_prob)
        self.dense2 = nn.Dense(reshaped_dim, out_dims)
        self.ac2 = nn.ReLU()

    def construct(self, x):
        x = self.bn1(x.transpose(0,2,1)).transpose(0,2,1)
        x = self.dense1(x)
        x = self.ac1(x)
        x = self.dp1(x)
        res_dim = ops.prod(ms.tensor(x.shape[1:])).item()
        x = self.dense2(x.reshape((-1,res_dim)))
        x = self.ac2(x)
        return x


class Critic(nn.Cell):
    def __init__(self, mlp_dim, in_dims, out_dims, keep_prob):
        super(Critic,self).__init__(auto_prefix=True)

        layers = list()
        layers.append(nn.BatchNorm1d(in_dims))
        layers.append(nn.Dense(in_dims, mlp_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(keep_prob=keep_prob))
        layers.append(nn.BatchNorm1d(mlp_dim))
        layers.append(nn.Dense(mlp_dim, out_dims))
        self.mlp = nn.SequentialCell(*layers)

    def construct(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class Actor(nn.Cell):
    def __init__(self, mlp_dim, in_dims, out_dims, keep_prob):
        super(Actor,self).__init__(auto_prefix=True)

        layers = list()
        layers.append(nn.BatchNorm1d(in_dims))
        layers.append(nn.Dense(in_dims, mlp_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(keep_prob=keep_prob))
        layers.append(nn.BatchNorm1d(mlp_dim))
        layers.append(nn.Dense(mlp_dim, out_dims))
        self.mlp = nn.SequentialCell(*layers)

        self.softmax = nn.Softmax(axis=-1)
        self.equal = ops.Equal()

    def construct(self, x, mask=None):
        out = self.mlp(x)
        if mask is not None:
            out = ops.where(self.equal(mask, 0), ops.zeros_like(out) - 100000.0, out)

        out_prob = self.softmax(out)
        return out_prob

class GatedAttention(nn.Cell):
    def __init__(self, mlp_dim, emb_dim, num_heads, keep_prob, seq_len):
        super(GatedAttention,self).__init__(auto_prefix=True)

        self.c1_att = nn.MultiheadAttention(embed_dim=mlp_dim,num_heads=num_heads,dropout=1.-keep_prob, batch_first=True)
        self.c2_att = nn.MultiheadAttention(embed_dim=mlp_dim,num_heads=num_heads,dropout=1.-keep_prob, batch_first=True)
        self.gated_mlp = nn.SequentialCell(
            nn.Dense(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(keep_prob=keep_prob),
            nn.Dense(mlp_dim, 2),
        )

        self.prj = nn.Dense(mlp_dim*seq_len, mlp_dim*3+emb_dim+1)
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim

    def construct(self, channel_state, user_hist, c1_cand, c2_cand):
        channel_state = channel_state.unsqueeze(1).repeat(self.seq_len,axis=1)
        c1_res,_ = self.c1_att(channel_state, c1_cand, c1_cand)
        c2_res,_ = self.c2_att(channel_state, c2_cand, c2_cand)
        gated_score = self.gated_mlp(user_hist)

        res = c1_res* gated_score[:,0:1].unsqueeze(2) + c2_res * gated_score[:,1:2].unsqueeze(2)
        res = self.prj(res.reshape((-1,self.seq_len*self.mlp_dim)))
        return res