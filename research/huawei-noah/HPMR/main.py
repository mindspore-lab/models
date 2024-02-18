import os
import sys
import pandas as pd
import copy
import random
from utility.helper import *
from utility.batch_test import *
import mindspore as ms
from mindspore import Tensor, SparseTensor, ops, nn, Parameter, context, COOTensor
from mindspore import ops as P
from mindspore.ops import functional as F
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, XavierNormal
import mindspore.numpy as mnp
from tqdm import tqdm
from mindspore import ParameterTuple


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp

class HPMR(nn.Cell):
    def __init__(self, max_item_view, max_item_cart, max_item_buy, max_item_pwc, max_item_pwb, max_item_cwb, data_config):
        super(HPMR, self).__init__()
        # argument settings
        self.model_type = 'HPMR'
        self.adj_type = args.adj_type
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 1
        self.wid=eval(args.wid)    # 0.1 for beibei, 0.01 for taobao
        self.buy_adj = data_config['buy_adj']
        self.pv_adj = data_config['pv_adj']
        self.cart_adj = data_config['cart_adj']
        self.pwc_adj = data_config['pwc_adj']
        self.pwb_adj = data_config['pwb_adj']
        self.cwb_adj = data_config['cwb_adj']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.gnn_layer
        self.decay = args.decay # 10 for beibei,1e-1 for taobao
        self.verbose = args.verbose
        self.max_item_view = max_item_view
        self.max_item_cart = max_item_cart
        self.max_item_buy = max_item_buy
        
        self.max_item_pwc = max_item_pwc
        self.max_item_pwb = max_item_pwb
        self.max_item_cwb = max_item_cwb      
        self.coefficient = eval(args.coefficient) # 0.0/6, 5.0/6,1.0/6 for beibei and 1.0/6, 4.0/6, 1.0/6 for taobao
        self.alpha = eval(args.alpha)
        self.n_relations=3
        self._init_weights()

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''



    def construct(self,data):
        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """
        # self.input_u, self.lable_buy, self.lable_view, self.lable_cart, self.lable_pwc, self.lable_pwb, self.lable_cwb, self.node_dropout, self.mess_dropout = input_u,lable_buy,lable_view,lable_cart,lable_pwc,lable_pwb,lable_cwb,node_dropout,mess_dropout
        self.input_u, self.lable_buy, self.lable_view, self.lable_cart, self.lable_pwc, self.lable_pwb, self.lable_cwb, self.node_dropout, self.mess_dropout, self.pos_items = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]
        self.ua_embeddings, self.ia_embeddings, self.r0, self.r1, self.r2, self.r_pwc, self.r_pwb, self.r_cwb = self._create_gcn_embed()

        """
        *********************************************************
        The module of the denoise.
        """
        
        # first_denoise
        self.ua_vc_clear, self.ua_vc_noise = self.denoise(self.ua_embeddings[0], self.ua_embeddings[1])
        self.ia_vc_clear, self.ia_vc_noise = self.denoise(self.ia_embeddings[0], self.ia_embeddings[1])
    
        self.ua_vb_clear, self.ua_vb_noise = self.denoise(self.ua_embeddings[0], self.ua_embeddings[2])
        self.ia_vb_clear, self.ia_vb_noise = self.denoise(self.ia_embeddings[0], self.ia_embeddings[2])
        
        self.ua_cb_clear, self.ua_cb_noise = self.denoise(self.ua_embeddings[1], self.ua_embeddings[2])
        self.ia_cb_clear, self.ia_cb_noise = self.denoise(self.ia_embeddings[1], self.ia_embeddings[2])

        
        # re-enhance_denoise
        self.ua_vc_clear_re, _ = self.denoise(self.ua_vc_noise, self.ua_embeddings[0])
        self.ia_vc_clear_re, _ = self.denoise(self.ia_vc_noise, self.ia_embeddings[0])
    
        self.ua_vb_clear_re, _ = self.denoise(self.ua_vb_noise, self.ua_embeddings[0])
        self.ia_vb_clear_re, _ = self.denoise(self.ia_vb_noise, self.ia_embeddings[0])
        
        self.ua_cb_clear_re, _ = self.denoise(self.ua_cb_noise, self.ua_embeddings[1])
        self.ia_cb_clear_re, _ = self.denoise(self.ia_cb_noise, self.ia_embeddings[1])

        self.ua_vc_clear_re *= args.re_mult
        self.ia_vc_clear_re *= args.re_mult
        self.ua_vb_clear_re *= args.re_mult
        self.ia_vb_clear_re *= args.re_mult
        self.ua_cb_clear_re *= args.re_mult
        self.ia_cb_clear_re *= args.re_mult
        
        """for training"""
        """embeddings for unique loss"""           

        self.ua_unique_embs = [self.ua_vc_noise, self.ua_vb_noise, self.ua_cb_noise]
        self.ia_unique_embs = [self.ia_vc_noise, self.ia_vb_noise, self.ia_cb_noise]
        
    
        """embeddings for re-enhance"""
        all_vc_noise_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_vc_clear_re, self.ia_vc_clear_re]), self.A_fold_hat_pv, args.transfer_gnn_layer, self.r0, 'clear')), 1)
        vc_noise_embs_user, vc_noise_embs_item = ops.split(all_vc_noise_embs, [self.n_users, self.n_items], 0)

        all_vb_noise_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_vb_clear_re, self.ia_vb_clear_re]), self.A_fold_hat_pv, args.transfer_gnn_layer, self.r0, 'clear')), 1)
        vb_noise_embs_user, vb_noise_embs_item = ops.split(all_vb_noise_embs, [self.n_users, self.n_items], 0)        
        
        all_cb_noise_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_cb_clear_re, self.ia_cb_clear_re]), self.A_fold_hat_cart, args.transfer_gnn_layer, self.r1, 'clear')), 1)
        cb_noise_embs_user, cb_noise_embs_item = ops.split(all_cb_noise_embs, [self.n_users, self.n_items], 0)        
        
        
        """embeddings for transfer"""
        all_vc_clear_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_vc_clear, self.ia_vc_clear]), self.A_fold_hat_cart, args.transfer_gnn_layer, self.r1, 'clear')), 1)
        vc_clear_embs_user, vc_clear_embs_item = ops.split(all_vc_clear_embs, [self.n_users, self.n_items], 0)

        all_vb_clear_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_vb_clear, self.ia_vb_clear]), self.A_fold_hat_buy, args.transfer_gnn_layer, self.r2, 'clear')), 1)
        vb_clear_embs_user, vb_clear_embs_item = ops.split(all_vb_clear_embs, [self.n_users, self.n_items], 0)  
        
        all_cb_clear_embs = ops.ReduceMean(keep_dims=False)(ops.Stack(axis=1)(self.gnns_transfer(ops.Concat(axis = 0)([self.ua_cb_clear, self.ia_cb_clear]), self.A_fold_hat_buy, args.transfer_gnn_layer, self.r2, 'clear')), 1)
        cb_clear_embs_user, cb_clear_embs_item = ops.split(all_cb_clear_embs, [self.n_users, self.n_items], 0)           

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        for test
        """
        self.users = ops.Squeeze()(Tensor(data[0],ms.int32))
        self.pos_items = Tensor(np.array(self.pos_items),ms.int32)
        self.u_g_embeddings = ops.gather(self.ua_embeddings[-1], self.users, 0)
        self.pos_i_g_embeddings = ops.gather(self.ia_embeddings[-1], self.pos_items, 0)
        
        # self.dot = ops.einsum('ac,bc->abc', self.u_g_embeddings, self.pos_i_g_embeddings)
        # self.batch_ratings = ops.einsum('ajk,lk->aj', self.dot, self.r2)   
        self.dot = ops.multiply(self.u_g_embeddings, self.r2) 
        self.batch_ratings = ops.MatMul(transpose_b=True)(self.dot, self.pos_i_g_embeddings)
           
        self.uid = []       
        self.input_u = ops.Squeeze()(Tensor(data[0],ms.int32))
        for idx, ua_embedding in enumerate(self.ua_embeddings):
            # View
            if idx == 0:
                uid_tmp = ops.gather((ua_embedding+vc_noise_embs_user+vb_noise_embs_user)/3, self.input_u, 0)
                uid_tmp = ops.Reshape()(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp)
            # Cart
            if idx == 1:
                uid_tmp = ops.gather((ua_embedding+vc_clear_embs_user+cb_noise_embs_user)/3, self.input_u, 0)
                uid_tmp = ops.Reshape()(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp)
            # Buy
            if idx == 2:
                uid_tmp = ops.gather((ua_embedding+vb_clear_embs_user+cb_clear_embs_user)/3, self.input_u, 0)
                uid_tmp = ops.Reshape()(uid_tmp, [-1, self.emb_dim])
                self.uid.append(uid_tmp) 
                    
        self.uid_main = []
        for idx, ua_embedding in enumerate(self.ua_embeddings):
                uid_tmp = ops.gather(ua_embedding, self.input_u, 0)
                uid_tmp = ops.Reshape()(uid_tmp, [-1, self.emb_dim])
                self.uid_main.append(uid_tmp)            
 
        self.noise_uid = []
        for idx, ua_embedding in enumerate(self.ua_unique_embs):
            uid_tmp = ops.gather(ua_embedding, self.input_u, 0)
            uid_tmp = ops.Reshape()(uid_tmp, [-1, self.emb_dim])
            self.noise_uid.append(uid_tmp)        
        
        # transfer predict
        self.pos_rv = self._get_pos_emb(self.ia_embeddings[0], self.lable_view, self.uid[0], self.r0)
        self.pos_rc = self._get_pos_emb(self.ia_embeddings[1], self.lable_cart, self.uid[1], self.r1)
        self.pos_rb = self._get_pos_emb(self.ia_embeddings[2], self.lable_buy, self.uid[2], self.r2)
        
        # main process predict
        self.pos_rv_main = self._get_pos_emb(self.ia_embeddings[0], self.lable_view, self.uid_main[0], self.r0)
        self.pos_rc_main = self._get_pos_emb(self.ia_embeddings[1], self.lable_cart, self.uid_main[1], self.r1)
        self.pos_rb_main = self._get_pos_emb(self.ia_embeddings[2], self.lable_buy, self.uid_main[2], self.r2)
        
        # unique predict
        self.pos_pwc_noise = self._get_pos_emb(self.ia_unique_embs[0], self.lable_pwc, self.noise_uid[0], self.r_pwc)
        self.pos_pwb_noise = self._get_pos_emb(self.ia_unique_embs[1], self.lable_pwb, self.noise_uid[1], self.r_pwb)
        self.pos_cwb_noise = self._get_pos_emb(self.ia_unique_embs[2], self.lable_cwb, self.noise_uid[2], self.r_cwb)

        self.pos_rs = [self.pos_rv, self.pos_rc, self.pos_rb]
        self.pos_mains = [self.pos_rv_main, self.pos_rc_main, self.pos_rb_main]
        self.rs = [self.r0, self.r1, self.r2]
        
        self.pos_beh_noise = [self.pos_pwc_noise, self.pos_pwb_noise, self.pos_cwb_noise]
        self.rs_noises = [self.r_pwc, self.r_pwb, self.r_cwb]
        output = [self.ia_embeddings, self.uid, self.uid_main, self.ia_unique_embs, self.noise_uid, self.pos_rs, self.pos_mains, self.rs, self.pos_beh_noise, self.rs_noises]
        return output
    
    def predict(self, users, pos_items, node_dropout=False, mess_dropout=False):
        if node_dropout:
            self.node_dropout = node_dropout
        if mess_dropout:
            self.mess_dropout = mess_dropout
        self.ua_embeddings, self.ia_embeddings, self.r0, self.r1, self.r2, self.r_pwc, self.r_pwb, self.r_cwb = self._create_gcn_embed()
        users = ops.Squeeze()(Tensor(users,ms.int32))
        pos_items = Tensor(np.array(pos_items),ms.int32)
        u_g_embeddings = ops.gather(self.ua_embeddings[-1], users, 0)
        pos_i_g_embeddings = ops.gather(self.ia_embeddings[-1], pos_items, 0)
        
        dot = ops.multiply(u_g_embeddings, self.r2) 
        batch_ratings = ops.MatMul(transpose_b=True)(dot, pos_i_g_embeddings)
        return batch_ratings
    
    def save_embedding(self, node_dropout, mess_dropout):
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.ua_embeddings, self.ia_embeddings, self.r0, self.r1, self.r2, self.r_pwc, self.r_pwb, self.r_cwb = self._create_gcn_embed()

    def _get_pos_emb(self, ia_embeddings, lable_beh, uid_beh, r):
        token_embedding =  ops.Zeros()((1, self.emb_dim), ms.float32)
        ia_embeddings = ops.Concat(axis = 0)([ia_embeddings, token_embedding])
        lable_beh = Tensor(lable_beh, ms.int32)
        pos_beh = ops.gather(ia_embeddings, lable_beh, 0)
        pos_num_beh = ops.Cast()(ops.not_equal(lable_beh, self.n_items), ms.float32)
        pos_beh = ops.multiply(ops.expand_dims(pos_num_beh,2),pos_beh)
        pos_beh = ops.multiply( ops.expand_dims(uid_beh,1), pos_beh)
        return ops.Squeeze()(ops.tensor_dot(pos_beh, r, ((2),(1))))

        # initialization of model parameters
    
    def _init_weights(self):
        # self.user_embedding = Parameter(initializer("xavier_uniform", [self.n_users, self.emb_dim], ms.float32), name='user_embedding', requires_grad=True)
        # self.item_embedding = Parameter(initializer("xavier_uniform", [self.n_items, self.emb_dim], ms.float32), name='item_embedding', requires_grad=True)
        self.all_embedding = Parameter(initializer("xavier_uniform", [self.n_users + self.n_items, self.emb_dim], ms.float32), name='all_embedding', requires_grad=True)
        self.relation_embedding = Parameter(initializer("xavier_uniform", [self.n_relations, self.emb_dim], ms.float32), name='relation_embedding', requires_grad=True)
        self.weight_size_list = [self.emb_dim] + [self.emb_dim] * max(self.n_layers, args.transfer_gnn_layer)

        self.all_weights = dict()
        for k in range(max(self.n_layers, args.transfer_gnn_layer)):
            self.all_weights['W_rel_%d%s' % (k,'clear')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_rel_%d%s' % (k,'clear'), requires_grad=True)
            
            self.all_weights['W_gc_%d%s' % (k,'clear')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_gc_%d%s' % (k,'clear'), requires_grad=True)
            self.all_weights['b_gc_%d%s' % (k,'clear')] = Parameter(
                initializer("xavier_uniform", [1, self.weight_size_list[k + 1]], ms.float32), name='b_gc_%d%s' % (k,'clear'), requires_grad=True)

            self.all_weights['W_bi_%d%s' % (k,'clear')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_bi_%d%s' % (k,'clear'), requires_grad=True)
            self.all_weights['b_bi_%d%s' % (k,'clear')] = Parameter(
                initializer("xavier_uniform", [1, self.weight_size_list[k + 1]], ms.float32), name='b_bi_%d%s' % (k,'clear'), requires_grad=True)

            self.all_weights['W_rel_%d%s' % (k,'noise')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_rel_%d%s' % (k,'noise'), requires_grad=True)
            
            self.all_weights['W_gc_%d%s' % (k,'noise')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_gc_%d%s' % (k,'noise'), requires_grad=True)
            self.all_weights['b_gc_%d%s' % (k,'noise')] = Parameter(
                initializer("xavier_uniform", [1, self.weight_size_list[k + 1]], ms.float32), name='b_gc_%d%s' % (k,'noise'), requires_grad=True)

            self.all_weights['W_bi_%d%s' % (k,'noise')] = Parameter(
                initializer("xavier_uniform", [self.weight_size_list[k], self.weight_size_list[k + 1]], ms.float32), name='W_bi_%d%s' % (k,'noise'), requires_grad=True)
            self.all_weights['b_bi_%d%s' % (k,'noise')] = Parameter(
                initializer("xavier_uniform", [1, self.weight_size_list[k + 1]], ms.float32), name='b_bi_%d%s' % (k,'noise'), requires_grad=True)

       
    def denoise(self, origin_emb, target_emb):
        res_array = ops.expand_dims(ops.ReduceSum(keep_dims=False)(ops.multiply(origin_emb,target_emb),axis=1),-1)*target_emb
        norm_num = ops.norm(target_emb, dim=1)*ops.norm(target_emb, dim=1)+1e-12
        clear_emb = res_array/ops.expand_dims(norm_num,-1)
        noise_emb = origin_emb - clear_emb
        if False:
            a = ops.Cast()(ops.ReduceSum(keep_dims=False)(ops.multiply(origin_emb,target_emb),axis=1)>=0, ms.float32)
            clear_emb *= ops.expand_dims(a,-1)
        return clear_emb*0.1, noise_emb*0.1     
    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(temp)

        return A_fold_hat

    def _split_A_hat_node_without_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(temp)

        return A_fold_hat

    def mess_drop(self, embs):
        return ops.dropout(embs,p=self.mess_dropout[0])

    def gnns_transfer(self, allEmbed, A_fold_hat, layers, r, flag):
        ego_embeddings = allEmbed
        all_embeddings = [ego_embeddings]
        all_r = [r]
        for index in range(layers):            
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(ops.SparseTensorDenseMatmul()(Tensor(A_fold_hat[f].indices,ms.int32), A_fold_hat[f].values, A_fold_hat[f].shape, all_embeddings[-1]))
            norm_embeddings = ops.Concat(axis = 0)(temp_embed)
            norm_embeddings = ops.multiply(norm_embeddings, r)
            if args.encoder == 'lightgcn':                
                lightgcn_embeddings = norm_embeddings
                # embeddings_tmp = self.mess_drop(lightgcn_embeddings)  
                embeddings_tmp = lightgcn_embeddings         
            elif args.encoder == 'gccf':
                gccf_embeddings = nn.LeakyReLU()(norm_embeddings)
                # embeddings_tmp = self.mess_drop(gccf_embeddings)
                embeddings_tmp = gccf_embeddings 
            r = ops.MatMul()(r, self.all_weights['W_rel_%d%s' % (index, flag)])        
            all_r.append(r)
            all_embeddings.append(embeddings_tmp)
        return all_embeddings

    def gnns(self, allEmbed, A_fold_hat, layers, r, flag):
        ego_embeddings = allEmbed
        all_embeddings = [ego_embeddings]
        all_r = [r]
        for index in range(layers):            
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(ops.SparseTensorDenseMatmul()(Tensor(A_fold_hat[f].indices,ms.int32),A_fold_hat[f].values,A_fold_hat[f].shape, all_embeddings[-1]))
            norm_embeddings = ops.Concat(axis=0)(temp_embed)
            norm_embeddings = ops.multiply(norm_embeddings, r)
            if args.encoder == 'lightgcn':                
                lightgcn_embeddings = norm_embeddings
                embeddings_tmp = self.mess_drop(lightgcn_embeddings)            
            elif args.encoder == 'gccf':
                gccf_embeddings = nn.LeakyReLU()(norm_embeddings)
                embeddings_tmp = self.mess_drop(gccf_embeddings)    
            r = ops.MatMul()(r, self.all_weights['W_rel_%d%s' % (index, flag)])        
            all_r.append(r)
            all_embeddings.append(embeddings_tmp)
        return all_embeddings, all_r
    
    def _create_gcn_embed(self):
        # node dropout.
        self.A_fold_hat_buy = self._split_A_hat_node_dropout(self.buy_adj)
        self.A_fold_hat_pv = self._split_A_hat_node_dropout(self.pv_adj)
        self.A_fold_hat_cart = self._split_A_hat_node_dropout(self.cart_adj)

        self.A_fold_hat_pwc = self._split_A_hat_node_dropout(self.pwc_adj)
        self.A_fold_hat_pwb = self._split_A_hat_node_dropout(self.pwb_adj)
        self.A_fold_hat_cwb = self._split_A_hat_node_dropout(self.cwb_adj)
        
        embeddings = self.all_embedding
        # embeddings = ops.Concat(axis = 0)([self.user_embedding, self.item_embedding])

        u_g_embeddings = []
        i_g_embeddings = []
        
        r0 = ops.gather(self.relation_embedding, Tensor(0, ms.int32), 0)
        r0 = ops.Reshape()(r0, [-1, self.emb_dim])

        r1 = ops.gather(self.relation_embedding, Tensor(1, ms.int32), 0)
        r1 = ops.Reshape()(r1, [-1, self.emb_dim])

        r2 = ops.gather(self.relation_embedding, Tensor(2, ms.int32), 0)
        r2 = ops.Reshape()(r2, [-1, self.emb_dim])
  
                      
        all_embeddings_pv, all_r0 = self.gnns(embeddings, self.A_fold_hat_pv, self.n_layers, r0, 'clear')
        all_embeddings_cart, all_r1 = self.gnns(embeddings, self.A_fold_hat_cart, self.n_layers, r1, 'clear')
        all_embeddings_buy, all_r2 = self.gnns(embeddings, self.A_fold_hat_buy, self.n_layers, r2, 'clear')

        _, all_pwc_noise = self.gnns(embeddings, self.A_fold_hat_pwc, self.n_layers, r0, 'noise')
        _, all_pwb_noise = self.gnns(embeddings, self.A_fold_hat_pwb, self.n_layers, r0, 'noise')
        _, all_cwb_noise = self.gnns(embeddings, self.A_fold_hat_cwb, self.n_layers, r1, 'noise')

        all_final_embs = [all_embeddings_pv, all_embeddings_cart, all_embeddings_buy]
        
        for idx, all_embedding in enumerate(all_final_embs):
            all_embedding = ops.AddN()(all_embeddings_pv)/len(all_embeddings_pv)
            u_g_embedding_tmp, i_g_embedding_tmp = ops.split(all_embedding, [self.n_users, self.n_items])
            u_g_embeddings.append(u_g_embedding_tmp)
            i_g_embeddings.append(i_g_embedding_tmp)        
        
        all_r0=ops.AddN()(all_r0)/len(all_r0)
        all_r1=ops.AddN()(all_r1)/len(all_r1)
        all_r2=ops.AddN()(all_r2)/len(all_r2)

        all_pwc_noise=ops.AddN()(all_pwc_noise)/len(all_pwc_noise)
        all_pwb_noise=ops.AddN()(all_pwb_noise)/len(all_pwb_noise)
        all_cwb_noise=ops.AddN()(all_cwb_noise)/len(all_cwb_noise)
        
        return u_g_embeddings, i_g_embeddings, all_r0, all_r1, all_r2, all_pwc_noise, all_pwb_noise, all_cwb_noise            


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        indices = Tensor(indices, dtype=ms.int32)
        data = Tensor(coo.data, dtype=ms.float32)
        return COOTensor(indices, data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += ms.Tensor(np.random.uniform(0, 1, noise_shape).astype(np.float32))
        dropout_mask = ops.Cast()(ops.Floor()(random_tensor), ms.bool_)
        pre_out = ops.MaskedSelect()(X, dropout_mask)

        return pre_out * ops.Div()(1., keep_prob)


def get_lables(temp_set,k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k)-1]

    # print max_item
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(view_lable, cart_lable, buy_lable, pwc_lable, pwb_lable, cwb_lable):
    user_train, view_item, cart_item, buy_item = [], [], [], []
    pwc_item, pwb_item, cwb_item = [], [], []
    for i in buy_lable.keys():
        user_train.append(i)
        buy_item.append(buy_lable[i])
        if i not in view_lable.keys():
            view_item.append([n_items] * max_item_view)
        else:
            view_item.append(view_lable[i])

        if i not in cart_lable.keys():
            cart_item.append([n_items] * max_item_cart)
        else:
            cart_item.append(cart_lable[i])
            
        if i not in pwc_lable.keys():
            pwc_item.append([n_items] * max_item_pwc)
        else:
            pwc_item.append(pwc_lable[i])
            
        if i not in pwb_lable.keys():
            pwb_item.append([n_items] * max_item_pwb)
        else:
            pwb_item.append(pwb_lable[i])
            
        if i not in cwb_lable.keys():
            cwb_item.append([n_items] * max_item_cwb)
        else:
            cwb_item.append(cwb_lable[i])

    user_train = np.array(user_train)
    view_item = np.array(view_item)
    cart_item = np.array(cart_item)
    buy_item = np.array(buy_item)

    pwc_item = np.array(pwc_item)
    pwb_item = np.array(pwb_item)
    cwb_item = np.array(cwb_item)
    
    user_train = user_train[:, np.newaxis]
    return user_train, view_item, cart_item, buy_item, pwc_item, pwb_item, cwb_item

class Loss_All(nn.LossBase):
    '''
    自定义loss函数
    '''
    def __init__(self,max_len=30):
        super(Loss_All, self).__init__("mean")
        self.coefficient = eval(args.coefficient) # 0.0/6, 5.0/6,1.0/6 for beibei and 1.0/6, 4.0/6, 1.0/6 for taobao
        self.alpha = eval(args.alpha)
        self.wid=eval(args.wid)
        self.decay = args.decay
 
    def construct(self, output,label):
        self.ia_embeddings = output[0]
        self.uid = output[1]
        self.uid_main = output[2]
        self.ia_unique_embs = output[3]
        self.noise_uid = output[4]
        temps = []
        temps_mains = []
        for idx, ia_embs in enumerate(self.ia_embeddings):
            temps.append(mnp.tensordot(ia_embs, ia_embs, axes=(0,0)) * mnp.tensordot(self.uid[idx], self.uid[idx], axes=(0,0)))
            temps_mains.append(mnp.tensordot(ia_embs, ia_embs, axes=(0,0)) * mnp.tensordot(self.uid_main[idx], self.uid_main[idx], axes=(0,0)))
            
        # the unique process
        temps_noise = []
        for idx, ia_embs in enumerate(self.ia_unique_embs):
            temps_noise.append(mnp.tensordot(ia_embs, ia_embs, axes=(0,0)) * mnp.tensordot(self.noise_uid[idx], self.noise_uid[idx], axes=(0,0))) 
                   
        loss1 = 0
        loss2 = 0
        loss3 = 0
        losses = [loss1, loss2, loss3]
        
        self.pos_rs = output[5]
        self.pos_mains = output[6]
        self.rs = output[7]
        
        self.pos_beh_noise = output[8]
        self.rs_noises = output[9]
        for idx in range(len(losses)):
            # the main process loss
            losses[idx] += self.alpha[0]*self.wid[idx]*ops.ReduceSum(keep_dims=False)(temps_mains[idx] * ops.MatMul(transpose_a=True)(self.rs[idx], self.rs[idx]))
            losses[idx] += self.alpha[0]*ops.ReduceSum(keep_dims=False)((1.0 - self.wid[idx]) * ops.Square()(self.pos_mains[idx]) - 2.0 * self.pos_mains[idx])
            
            # the transfer loss
            losses[idx] += self.alpha[1]*self.wid[idx]*ops.ReduceSum(keep_dims=False)(temps[idx] * ops.MatMul(transpose_a=True)(self.rs[idx], self.rs[idx]))
            losses[idx] += self.alpha[1]*ops.ReduceSum(keep_dims=False)((1.0 - self.wid[idx]) * ops.Square()(self.pos_rs[idx]) - 2.0 * self.pos_rs[idx])

            # the unique loss of cart
            if idx == 2:
                losses[1] += self.alpha[2]*self.wid[idx]*ops.ReduceSum(keep_dims=False)(temps_noise[idx] * ops.MatMul(transpose_a=True)(self.rs_noises[idx], self.rs_noises[idx]))
                losses[1] += self.alpha[2]*ops.ReduceSum(keep_dims=False)((1.0 - self.wid[idx]) * ops.Square()(self.pos_beh_noise[idx]) - 2.0 * self.pos_beh_noise[idx])  
            # the unique loss of view
            else:
                losses[0] += self.alpha[2]*self.wid[idx]*ops.ReduceSum(keep_dims=False)(temps_noise[idx] * ops.MatMul(transpose_a=True)(self.rs_noises[idx], self.rs_noises[idx]))
                losses[0] += self.alpha[2]*ops.ReduceSum(keep_dims=False)((1.0 - self.wid[idx]) * ops.Square()(self.pos_beh_noise[idx]) - 2.0 * self.pos_beh_noise[idx])                               
      

        loss = self.coefficient[0] * losses[0] + self.coefficient[1] * losses[1] + self.coefficient[2] * losses[2]         
        regularizer = self.alpha[0]*(ops.L2Loss()(self.uid_main[-1])) +\
            self.alpha[1]*(ops.L2Loss()(self.uid[-1]) + ops.L2Loss()(self.ia_embeddings[-1])) +\
            self.alpha[2]*(ops.L2Loss()(self.noise_uid[-1])+ops.L2Loss()(self.ia_unique_embs[-1]))        

        emb_loss = self.decay * regularizer
        batch_loss = emb_loss + loss

        return batch_loss

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, value_batch, label_batch):
        logits = self._backbone(value_batch)
        loss = self._loss_fn(logits, label_batch)
        return loss

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)
        self.grad_reducer = F.identity

    def construct(self, value_batch, label_batch):
        weights = self.weights
        loss = self.network(value_batch, label_batch)
        grads = self.grad(self.network, weights)(value_batch, label_batch)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

if __name__ == '__main__':
    print('now init')
    random.seed(42)
    np.random.seed(42)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=args.gpu_id)

    log_dir = 'log/' + args.dataset + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    import datetime
    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text
    
    from print_hook import PrintHook
    ph_out = PrintHook()
    ph_out.Start(my_hook_out)
    
    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))
 
    # logger.saveDefault = True   
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
    pre_adj,pre_adj_pv,pre_adj_cart,pre_adj_mat_pwc,pre_adj_mat_pwb,pre_adj_mat_cwb = data_generator.get_adj_mat() #

    config['buy_adj'] = pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    config['pwc_adj'] = pre_adj_mat_pwc
    config['pwb_adj'] = pre_adj_mat_pwb
    config['cwb_adj'] = pre_adj_mat_cwb
    print('use the pre adjcency matrix')

    n_users, n_items = data_generator.n_users, data_generator.n_items

    train_items = np.load(data_generator.path + '/train_items.npy', allow_pickle='TRUE').item()
    pv_set = np.load(data_generator.path + '/pv_set.npy', allow_pickle='TRUE').item()
    cart_set = np.load(data_generator.path + '/cart_set.npy', allow_pickle='TRUE').item()

    pv_wo_cart_set = np.load(data_generator.path + '/pv_wo_cart.npy', allow_pickle='TRUE').item()
    pv_wo_buy_set = np.load(data_generator.path + '/pv_wo_buy.npy', allow_pickle='TRUE').item()
    cart_wo_buy_set = np.load(data_generator.path + '/cart_wo_buy.npy', allow_pickle='TRUE').item()
    
    max_item_buy, buy_lable = get_lables(train_items)
    max_item_view, view_lable = get_lables(pv_set)
    max_item_cart, cart_lable = get_lables(cart_set)

    max_item_pwc, pwc_lable = get_lables(pv_wo_cart_set)
    max_item_pwb, pwb_lable = get_lables(pv_wo_buy_set)
    max_item_cwb, cwb_lable = get_lables(cart_wo_buy_set)
    
    t0 = time()
    print('model initial')
    model = HPMR(max_item_view, max_item_cart, max_item_buy, max_item_pwc, max_item_pwb, max_item_cwb, data_config=config)
    opt = nn.Adam(model.trainable_params(), learning_rate=args.lr)
    loss = Loss_All()
    loss_net = CustomWithLossCell(model, loss)
    train_net = TrainOneStepCell(loss_net, opt)

    for m in model.parameters_and_names():
        print('all the parameters',m)
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, view_item1, cart_item1, buy_item1, pwc_item1, pwb_item1, cwb_item1 = get_train_instances1(view_lable, cart_lable, buy_lable, pwc_lable, pwb_lable, cwb_lable)

    best_hr = 0
    base_save_path = './checkpoints/'+str(os.path.basename(__file__).split(".")[0])+'/'+args.dataset
    for epoch in range(args.epoch):

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        view_item1 = view_item1[shuffle_indices]
        cart_item1 = cart_item1[shuffle_indices]
        buy_item1 = buy_item1[shuffle_indices]
        pwc_item1 = pwc_item1[shuffle_indices]
        pwb_item1 = pwb_item1[shuffle_indices]
        cwb_item1 = cwb_item1[shuffle_indices]

        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size)

        for idx in tqdm(range(n_batch)):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            v_batch = view_item1[start_index:end_index]
            c_batch = cart_item1[start_index:end_index]
            b_batch = buy_item1[start_index:end_index]

            pwc_batch = pwc_item1[start_index:end_index]
            pwb_batch = pwb_item1[start_index:end_index]
            cwb_batch = cwb_item1[start_index:end_index]

            item_batch = range(0, 256)
            data = [u_batch,b_batch,v_batch,c_batch,pwc_batch,pwb_batch,cwb_batch,eval(args.node_dropout),eval(args.mess_dropout),item_batch]
            label= [b_batch,v_batch,c_batch,pwc_batch,pwb_batch,cwb_batch]
            train_net.set_train()
            # batch_mf_loss, batch_emb_loss = train_net(data,label)
            # batch_loss = batch_mf_loss + batch_emb_loss
            batch_loss = train_net(data,label)
            loss += batch_loss / n_batch
            # mf_loss += batch_mf_loss / n_batch
            # emb_loss += batch_emb_loss / n_batch

        if ops.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 2 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test( model, users_to_test, drop_flag=True)
        if args.save_emb == 1:
            if ret['hit_ratio'][0] > best_hr:
                specific_fname = '/'+'final'+str(epoch)+'_'+str(args.encoder)
                best_hr = ret['hit_ratio'][0]
                if os.path.exists(base_save_path) != True:
                    os.makedirs(base_save_path)
                tmp_user,tmp_item = model.save_embedding([0.] * args.gnn_layer,[0.] * args.gnn_layer)
                                                                                
                best_fname = base_save_path+specific_fname       
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f], ' \
                       'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0],
                           ret['precision'][0], ret['hit_ratio'][0],
                           ret['ndcg'][0])
            print(perf_str)

            """
            *********************************************************
            Get the performance w.r.t. different sparsity levels.
            """
            if 0:
                users_to_test_list, split_state = data_generator.get_sparsity_split()

                for i, users_to_test in enumerate(users_to_test_list):
                    ret = test( model, users_to_test, drop_flag=True)

                    final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                    print(final_perf)
        

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            if args.save_emb == 1:
                best_hr = ret['hit_ratio'][0]
                print('best_hr =', best_hr)
                if os.path.exists(base_save_path) != True:
                    os.makedirs(base_save_path)
                np.save(best_fname+'_user.npy',tmp_user)
                np.save(best_fname+'_item.npy',tmp_item)
                print('Best user & item embeddings are saved!')                                                                         
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)





