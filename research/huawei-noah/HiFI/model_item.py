import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform
import numpy as np
from model import Actor, Critic, StateMLP


class HIFII(nn.Cell):
    def __init__(self, params):
        super(HIFII, self).__init__(auto_prefix=True)
        self.params=params
        self.seq_len = params['seq_len']
        self.user_hist_len = params['user_hist_len']
        self.item_nums = params['item_nums']
        self.channel_nums = params['channel_nums']
        self.dense_dim = params['dense_dim']
        self.user_nums = params['user_nums']
        self.keep_prob = params['keep_prob']
        self.gamma = params['gamma']
        self.crr_type = params['crr_type']
        self.crr_ratio_upper_bound = 20.
        self.critic_weight = params['critic_weight']

        self.emb_dim = params['emb_dim']
        self.mlp_dim = params['mlp_dim']

        # id embeddings
        self.item_id_emb = nn.Embedding(vocab_size=self.item_nums, embedding_size=self.emb_dim)
        self.channel_id_emb = nn.Embedding(vocab_size=self.channel_nums, embedding_size=self.emb_dim)
        self.pos_emb = nn.Embedding(vocab_size=self.seq_len, embedding_size=self.emb_dim)

        # networks
        ## states
        self.hist_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim*2+self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.user_hist_len,keep_prob=self.keep_prob)
        self.cand_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim * 2 + self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.seq_len,keep_prob=self.keep_prob)
        self.rank_mlp = StateMLP(mlp_dim=self.mlp_dim, in_dims=self.emb_dim * 2 + self.dense_dim, out_dims=self.mlp_dim,
                                 reshaped_dim=self.mlp_dim*self.seq_len,keep_prob=self.keep_prob)

        self.actor_predict = Actor(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim*2, out_dims=self.seq_len-1, keep_prob=self.keep_prob)
        self.critic_predict = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim*2, out_dims=self.seq_len-1, keep_prob=self.keep_prob)
        self.actor_target = Actor(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim*2, out_dims=self.seq_len-1, keep_prob=self.keep_prob)
        self.critic_target = Critic(mlp_dim=self.mlp_dim, in_dims=self.mlp_dim*3+self.emb_dim*2, out_dims=self.seq_len-1, keep_prob=self.keep_prob)

        for params in self.actor_target.trainable_params():
            params.requires_grad=False
        for params in self.critic_target.trainable_params():
            params.requires_grad=False

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
        cur_ch, next_ch = all_states[5][:,:-1].reshape((-1,1)), all_states[5][:,1:].reshape((-1,1))

        merged_cur_state = self.build_data_embeddings(cur_hist, cur_cand, cur_ranked, cur_pos, cur_ch)
        merged_next_state = self.build_data_embeddings(cur_hist, next_cand, next_ranked, next_pos, next_ch)

        # =====learn critic=====
        q_value = self.critic_predict(merged_cur_state)
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

        critic_loss = ops.mean(diff.square())

        # =====learn actor=====
        pi = self.actor_predict(merged_cur_state)
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

        return loss, pred

    def build_data_embeddings(self, hist, cand, ranked, pos, ch, is_test=False):
        ch = ch.astype("int32")
        hist_id_emb = self.item_id_emb(hist[:,:,0].astype("int32"))
        hist_ch_emb = self.channel_id_emb(hist[:,:,1].astype("int32"))
        cand_id_emb = self.item_id_emb(cand[ops.arange(cand.shape[0]),ch.flatten(),:,0].astype("int32"))
        cand_ch_emb = self.channel_id_emb(cand[ops.arange(cand.shape[0]),ch.flatten(),:,1].astype("int32"))
        ranked_id_emb = self.item_id_emb(ranked[:,:,0].astype("int32"))
        ranked_ch_emb = self.channel_id_emb(ranked[:,:,1].astype("int32"))
        pos_emb = self.pos_emb(pos.astype("int32"))
        cur_ch_emb = self.channel_id_emb(ch)

        hist_emb = ops.cat([hist_id_emb, hist_ch_emb, hist[:,:,2:].astype("float32")], axis=-1)
        cand_emb = ops.cat([cand_id_emb, cand_ch_emb, cand[ops.arange(cand.shape[0]),ch.flatten(),:,2:].astype("float32")], axis=-1)
        ranked_emb = ops.cat([ranked_id_emb, ranked_ch_emb, ranked[:,:,2:].astype("float32")], axis=-1)

        merged_state = self.build_input_states(hist_emb, cand_emb, ranked_emb, pos_emb, cur_ch_emb, is_test)

        return merged_state

    def build_input_states(self, hist_emb, cand_emb, ranked_emb, pos_emb, ch_feat, is_test=False):
        hist_state = self.hist_mlp(hist_emb)
        cand_state = self.cand_mlp(cand_emb.reshape((-1,self.seq_len,self.emb_dim*2+self.dense_dim)))
        ranked_state = self.rank_mlp(ranked_emb)
        if is_test:
            hist_state = hist_state.reshape((-1, self.mlp_dim))
        else:
            hist_state = hist_state.unsqueeze(1).repeat(self.seq_len-1,axis=1).reshape((-1,self.mlp_dim))
        merged_state = ops.cat(
            [hist_state, cand_state, ranked_state, pos_emb.reshape((-1,self.emb_dim)), ch_feat.reshape((-1,self.emb_dim))], axis=-1)
        return merged_state

    def infer(self, all_states):

        # build current state and next state
        cur_hist = all_states[0].reshape((-1,self.user_hist_len,2+self.dense_dim))
        cur_cand = all_states[1].reshape((-1,self.channel_nums, self.seq_len, 2+self.dense_dim))
        cur_ranked = all_states[2].reshape((-1,self.seq_len, 2+self.dense_dim))
        cur_pos = all_states[3].reshape((-1,1))
        cur_ch = all_states[5].reshape((-1,1))

        merged_cur_state = self.build_data_embeddings(cur_hist, cur_cand, cur_ranked, cur_pos, cur_ch, is_test=True)

        # =====learn actor=====
        pi = self.actor_predict(merged_cur_state)

        pred = ops.argmax(pi, dim=-1)

        return pred, pi
