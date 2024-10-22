import numpy as np
import pandas as pd
import os
import ast
import collections


def cal_js(c0_num, c1_num, target_dist=[0.5,0.5]):
    eps = 1.e-6
    c0_dist = c0_num / (c0_num + c1_num+eps)
    c1_dist = c1_num / (c0_num + c1_num+eps)
    # print(c0_dist,c1_dist,other_dist)
    left_kl = c0_dist * np.log(c0_dist / ((target_dist[0] + c0_dist) / 2+eps)+eps) + c1_dist * np.log(
        c1_dist / ((target_dist[1] + c1_dist) / 2+eps)+eps)
    right_kl = target_dist[0] * np.log(target_dist[0] / ((target_dist[0] + c0_dist) / 2+eps)+eps) + target_dist[1] * np.log(
        target_dist[1] / ((target_dist[1] + c1_dist) / 2+eps)+eps)
    js = (left_kl + right_kl) / 2.
    return js


class HIFIDataset(object):
    def __init__(self, data_dir='data/', params=None):
        self.params = params
        self.user_feats, self.item_feats, self.inter_seqs = self.read_data(data_dir)

        self.all_state, self.batch_channel_action, self.batch_item_action, \
            self.batch_channel_reward, self.batch_item_reward, \
            self.batch_done = self.construct_state()
        self.state_key = ['history', 'candidates', 'ranked_items', 'pos', 'js_value', 'selected_ch']

    def __len__(self):
        return self.batch_done.shape[0]

    def __getitem__(self, index):
        return self.all_state[self.state_key[0]][index], self.all_state[self.state_key[1]][index], \
                  self.all_state[self.state_key[2]][index], self.all_state[self.state_key[3]][index], \
                    self.all_state[self.state_key[4]][index], self.all_state[self.state_key[5]][index], \
                    self.batch_channel_action[index], self.batch_item_action[index], \
                    self.batch_channel_reward[index], self.batch_item_reward[index], \
                    self.batch_done[index]

    def read_data(self, data_dir):
        user_feat_df = pd.read_csv(os.path.join(data_dir, 'user_feat.csv'))
        user_feat_df['user_hist'] = user_feat_df['user_hist'].apply(ast.literal_eval)
        user_feats = user_feat_df.values.tolist()

        item_feat_df = pd.read_csv(os.path.join(data_dir, 'item_feat.csv'))
        item_feats = item_feat_df.values.tolist()

        inter_seq_df = pd.read_csv(os.path.join(data_dir, 'inter_seq.csv'))
        inter_seq_df['items'] = inter_seq_df['items'].apply(ast.literal_eval)
        inter_seq_df['click'] = inter_seq_df['click'].apply(ast.literal_eval)
        inter_seqs = inter_seq_df.values.tolist()

        user_feats = {int(oneline[0]): oneline[1] for oneline in user_feats}
        item_feats = {int(oneline[0]): oneline[1:] for oneline in item_feats}

        return user_feats, item_feats, inter_seqs

    def construct_state(self):
        state = {'history': [], 'candidates': [], 'ranked_items': [], 'pos': [],
                 'js_value': [], 'selected_ch': []}
        reward = [[], []]
        action = [[], []]
        done = []

        for idxxx in range(len(self.inter_seqs)):
            user_id, items, click = self.inter_seqs[idxxx]
            user_id = int(user_id)
            items = np.array(items, dtype=int)
            click = np.array(click, dtype=int)

            # build history array, [id, channel, dense]*hist_len
            history_features = np.zeros((self.params['user_hist_len'], 1+1 + self.params['dense_dim']))
            for i, item_id in enumerate(self.user_feats[int(user_id)]):
                history_features[i,0] = item_id
                history_features[i,1:] = self.item_feats[item_id]

            # build candidates for 2 channel

            candidate_features = np.zeros((self.params['seq_len'], 2+self.params['dense_dim']))
            for i, item_id in enumerate(items):
                candidate_features[i,0] = int(item_id)
                candidate_features[i,1:] = self.item_feats[int(item_id)]

            cats = candidate_features[:, 1].astype(int)

            c0_feat = candidate_features[np.where(cats == 0)]
            c1_feat = candidate_features[np.where(cats == 1)]

            # build 2 channels' candidate items, ranked items,  pos, and js values

            cur_candidates = np.zeros(
                (self.params['channel_nums'], self.params['seq_len'], 2+self.params['dense_dim']))
            cur_ranked = np.zeros((self.params['seq_len'], self.params['seq_len'], 2+self.params['dense_dim']))
            cur_pos = np.zeros(self.params['seq_len'])
            cur_js = np.zeros(self.params['seq_len'])

            cur_candidates[0, :c0_feat.shape[0], :] = c0_feat
            cur_candidates[1, :c1_feat.shape[0], :] = c1_feat
            cur_candidates = np.repeat(np.expand_dims(cur_candidates, axis=0), self.params['seq_len'], axis=0)

            used_channel_num = [0, 0]
            for time_step in range(1,self.params['seq_len']):
                cnts = collections.Counter(cats[:time_step])
                cur_ranked_channel = int(cats[time_step])
                cur_candidates[time_step:,cur_ranked_channel,used_channel_num[cur_ranked_channel],:]=0
                cur_ranked[time_step,:time_step,:]=candidate_features[:time_step,:]
                cur_pos[time_step]=time_step
                if time_step>=self.params['seq_len']//2:
                    cur_js[time_step] = cal_js(cnts[0],cnts[1])
                else:
                    cur_js[time_step]=0.
                used_channel_num[cur_ranked_channel]+=1

            # state len is t, :t-1 is cur state, and 1: is next state
            state['history'].append(history_features)
            state['candidates'].append(cur_candidates)
            state['ranked_items'].append(cur_ranked)
            state['pos'].append(cur_pos)
            state['js_value'].append(cur_js)
            state['selected_ch'].append(cats[:self.params['seq_len']])

            # action and reward
            cur_c_action, cur_i_action = np.zeros(self.params['seq_len']-1), np.zeros(self.params['seq_len']-1)
            cur_c_reward, cur_i_reward = np.zeros(self.params['seq_len']-1), np.zeros(self.params['seq_len']-1)  # channel r, item r

            news_clicks = click[np.where(cats == 0)]
            news_clicks = np.where(news_clicks != 1, np.zeros_like(news_clicks), news_clicks)
            sports_clicks = click[np.where(cats == 1)]
            sports_clicks = np.where(sports_clicks != 1, np.zeros_like(sports_clicks), sports_clicks)
            cs = [news_clicks, sports_clicks]

            used_channel_num = [0, 0, 0]

            for idx in range(self.params['seq_len']-1):
                cur_cat = int(cats[idx])
                if cur_cat not in [0, 1, 2]:
                    cur_cat = 0
                cur_c_action[idx] = cur_cat  # selected channels idx
                cur_c_reward[idx] = np.sum(cs[cur_cat][used_channel_num[cur_cat]:])
                cur_i_action[idx] = used_channel_num[cur_cat]  # selected items pos
                cur_i_reward[idx] = click[idx]

                used_channel_num[cur_cat] += 1

            action[0].append(cur_c_action)
            action[1].append(cur_i_action)
            reward[0].append(cur_c_reward)
            reward[1].append(cur_i_reward)

            # stat done signal
            cur_done = np.zeros(self.params['seq_len']-1)
            cur_done[-1] = 1
            done.append(cur_done)

        k = ['history', 'candidates', 'ranked_items', 'pos', 'js_value', 'selected_ch']
        all_state = {}
        for o_k in k:
            all_state[o_k]=np.stack(state[o_k], axis=0)
        batch_channel_action = np.stack(action[0], axis=0)
        batch_item_action = np.stack(action[1], axis=0)

        batch_channel_reward = np.stack(reward[0], axis=0)
        batch_item_reward = np.stack(reward[1], axis=0)

        batch_done = np.stack(done, axis=0)

        return all_state, batch_channel_action, batch_item_action, batch_channel_reward, batch_item_reward, batch_done

class EvalDataset(object):
    def __init__(self, data_dir='data/', params=None):
        self.params = params
        self.user_feats, self.item_feats, self.inter_seqs = self.read_data(data_dir)

        self.all_state, self.click_list_0, self.click_list_1, = self.construct_state()
        self.state_key = ['history', 'candidates', 'ranked_items', 'pos', 'js_value', 'selected_ch']

    def __len__(self):
        return self.click_list_0.shape[0]

    def __getitem__(self, index):
        return self.all_state[self.state_key[0]][index], self.all_state[self.state_key[1]][index], \
            self.all_state[self.state_key[2]][index], self.all_state[self.state_key[3]][index], \
            self.all_state[self.state_key[4]][index], self.all_state[self.state_key[5]][index], \
            self.click_list_0[index], self.click_list_1[index]

    def read_data(self, data_dir):
        user_feat_df = pd.read_csv(os.path.join(data_dir, 'user_feat.csv'))
        user_feat_df['user_hist'] = user_feat_df['user_hist'].apply(ast.literal_eval)
        user_feats = user_feat_df.values.tolist()

        item_feat_df = pd.read_csv(os.path.join(data_dir, 'item_feat.csv'))
        item_feats = item_feat_df.values.tolist()

        inter_seq_df = pd.read_csv(os.path.join(data_dir, 'test_inter_seq.csv'))
        inter_seq_df['items'] = inter_seq_df['items'].apply(ast.literal_eval)
        inter_seq_df['click'] = inter_seq_df['click'].apply(ast.literal_eval)
        inter_seqs = inter_seq_df.values.tolist()

        user_feats = {int(oneline[0]): oneline[1] for oneline in user_feats}
        item_feats = {int(oneline[0]): oneline[1:] for oneline in item_feats}

        return user_feats, item_feats, inter_seqs

    def construct_state(self):
        state = {'history': [], 'candidates': [], 'ranked_items': [], 'pos': [],
                 'js_value': [], 'selected_ch': []}
        clicks_lists = [[], []]

        for idxxx in range(len(self.inter_seqs)):
            user_id, items, click = self.inter_seqs[idxxx]
            user_id = int(user_id)
            items = np.array(items, dtype=int)
            click = np.array(click, dtype=int)

            # build history array, [id, channel, dense]*hist_len
            history_features = np.zeros((self.params['user_hist_len'], 1 + 1 + self.params['dense_dim']))
            for i, item_id in enumerate(self.user_feats[int(user_id)]):
                history_features[i, 0] = item_id
                history_features[i, 1:] = self.item_feats[item_id]

            # build candidates for 2 channel

            candidate_features = np.zeros((self.params['seq_len'], 2 + self.params['dense_dim']))
            for i, item_id in enumerate(items):
                candidate_features[i, 0] = int(item_id)
                candidate_features[i, 1:] = self.item_feats[int(item_id)]

            cats = candidate_features[:, 1].astype(int)

            c0_feat = candidate_features[np.where(cats == 0)]
            c1_feat = candidate_features[np.where(cats == 1)]

            # build 2 channels' candidate items, ranked items,  pos, and js values

            cur_candidates = np.zeros(
                (self.params['channel_nums'], self.params['seq_len'], 2 + self.params['dense_dim']))
            cur_ranked = np.zeros((self.params['seq_len'], 2 + self.params['dense_dim']))
            cur_pos = np.zeros(1)
            cur_js = np.zeros(1)

            cur_candidates[0, :c0_feat.shape[0], :] = c0_feat
            cur_candidates[1, :c1_feat.shape[0], :] = c1_feat

            state['history'].append(history_features)
            state['candidates'].append(cur_candidates)
            state['ranked_items'].append(cur_ranked)
            state['pos'].append(cur_pos)
            state['js_value'].append(cur_js)
            state['selected_ch'].append(np.zeros(1))

            news_clicks = click[np.where(cats == 0)]
            reshaped_news_clicks = np.zeros(self.params['seq_len'])
            reshaped_news_clicks[:len(news_clicks)] = news_clicks[:len(news_clicks)]
            reshaped_news_clicks = np.where(reshaped_news_clicks != 1, np.zeros_like(reshaped_news_clicks),
                                            reshaped_news_clicks)
            sports_clicks = click[np.where(cats == 1)]
            reshaped_sports_clicks = np.zeros(self.params['seq_len'])
            reshaped_sports_clicks[:len(sports_clicks)] = sports_clicks[:len(sports_clicks)]
            reshaped_sports_clicks = np.where(reshaped_sports_clicks != 1, np.zeros_like(reshaped_sports_clicks),
                                              reshaped_sports_clicks)

            clicks_lists[0].append(reshaped_news_clicks)
            clicks_lists[1].append(reshaped_sports_clicks)

        k = ['history', 'candidates', 'ranked_items', 'pos', 'js_value', 'selected_ch']
        all_state = {}
        for o_k in k:
            all_state[o_k] = np.stack(state[o_k], axis=0)

        click_list_0 = np.stack(clicks_lists[0], axis=0)
        click_list_1 = np.stack(clicks_lists[1], axis=0)

        return all_state, click_list_0, click_list_1
