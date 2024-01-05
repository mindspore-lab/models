import mindspore as ms
import mindspore.nn as nn
import numpy as np

from .mixed_op import MixedOp
from .mode import Mode


class Cell(nn.Cell):
    def __init__(self, channels, num_nodes, candidate_op_profiles):
        super(Cell, self).__init__()
        # create mixed operations
        self._channels = channels
        self._num_nodes = num_nodes
        self._mixed_ops = nn.CellList()
        self._mixed_ops_ids = {}#nn.ModuleDict()
        self.cnt = 0

        for i in range(1, self._num_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i,j)
                mop = MixedOp(candidate_op_profiles)
                mop.update_parameters_name(prefix=f'Node{i}_{j}', recurse=True)
                self._mixed_ops_ids[node_str] = self.cnt
                self.cnt = self.cnt + 1
                self._mixed_ops.append(mop)

        self._edge_keys = sorted(list(self._mixed_ops_ids.keys()))
        self._edge2index = {key:i for i,key in enumerate(self._edge_keys)}
        self._index2edge = {i:key for i,key in enumerate(self._edge_keys)}
        self._num_edges = len(self._mixed_ops_ids) # num_mixed_ops
        self._num_ops = len(candidate_op_profiles) # number of differnet op types

        # arch_weights
        self._candidate_alphas = ms.Parameter(1e-3*ms.ops.randn(self._num_edges, self._num_ops), requires_grad=True) # [num_edges, num_ops]
        # pt_weights
        self._candidate_flags = ms.Parameter(ms.Tensor(self._num_edges*[True], dtype=ms.bool_), requires_grad=False) # # [num_edges,]
        self._project_weights = ms.Parameter(ms.ops.zeros_like(self._candidate_alphas), requires_grad=False) # [num_edges, num_ops]

        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        self.set_edge_mode(mode)

    def set_edge_mode(self, mode=None):
        mode = mode or self._mode
        edge_sample_idx = {}
        for key, mix_op_id in self._mixed_ops_ids.items(): # loop all edges, key: edge_name
            mix_op = self._mixed_ops[mix_op_id]
            if mode == Mode.PROJECT:
                if self._candidate_flags[self._edge2index[key]]==False:
                    op_id = ms.ops.argmax(self._project_weights[self._edge2index[key]]).asnumpy().item() # choose the best op to set path
                    sample_idx = np.array([op_id], dtype=np.int32)
                else:
                    sample_idx = np.arange(self._num_ops)
            elif mode == Mode.NONE:
                sample_idx = None
            elif mode == Mode.ONE_PATH_FIXED:
                probs = ms.ops.softmax(ms.ops.stop_gradient(self._candidate_alphas[self._edge2index[key]]), axis=0)
                op = ms.ops.argmax(probs).asnumpy().item()
                sample_idx = np.array([op], dtype=np.int32)
            elif mode == Mode.ONE_PATH_RANDOM:
                probs = ms.ops.softmax(ms.ops.stop_gradient(self._candidate_alphas[self._edge2index[key]]), axis=0)
                sample_idx = ms.ops.multinomial(probs, 1, replacement=True).asnumpy()
            elif mode == Mode.TWO_PATHS:
                probs = ms.ops.softmax(ms.ops.stop_gradient(self._candidate_alphas[self._edge2index[key]]), axis=0)
                sample_idx = ms.ops.multinomial(probs, 2, replacement=True).asnumpy()
            elif mode == Mode.ALL_PATHS:
                sample_idx = np.arange(self._num_ops)
            else:
                sample_idx = np.arange(self._num_ops)

            mix_op.set_mode(mode, sample_idx)
            edge_sample_idx[key] = sample_idx
        return edge_sample_idx

    def project_op(self, e_id, op_id):
        # set best op of the edge to 1
        self._project_weights[e_id][op_id] = 1 ## hard by default
        # if this edge had been choosed, set it to False
        self._candidate_flags[e_id] = False

    def project_parameters(self):
        # using arch weights get by DARTS search
        weights = ms.ops.softmax(self._candidate_alphas, axis=-1)
        for e_id in range(self._num_edges):
            if not self._candidate_flags[e_id]: # if the edge had been choosed, set new weights using hard code (1,0)
                weights[e_id].data.copy_(self._project_weights[e_id])

        return weights

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for key, mix_op_id in self._mixed_ops_ids.items():
            for name, p in self._mixed_ops[mix_op_id].parameters_and_names():
                yield p
    
    def proj_parameters(self):
        for pt_weight in [self._candidate_flags, self._project_weights]:
            yield pt_weight

    def num_weight_parameters(self):
        count = 0
        for key, mix_op_id in self._mixed_ops_ids.items():
            count += self._mixed_ops[mix_op_id].num_weight_parameters()
        return count

    def construct(self, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class ALLOTCell(Cell):
    def __init__(self, channels, num_mixed_ops, candidate_op_profiles):
        super(ALLOTCell, self).__init__(channels, num_mixed_ops, candidate_op_profiles)

        # self.norm_layer1 = nn.LayerNorm(channels)
        # self.norm_layer2 = nn.LayerNorm(channels)
        
    def ld(self, name):
        return ms.Tensor(np.load("../../Code_ALLOT/src/" + name + ".npy"), ms.float32)
    
    def construct(self, x, x_mark, attn_mask=None, adj_mats=None, weights=None, **kwargs):
        # num_mixed_ops = 1
        nodes = [[x, kwargs['st']]]
        edge_sample_idx = self.set_edge_mode(self._mode)
        for i in range(1, self._num_nodes):
            inter_nodes_tt = []
            inter_nodes_st = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i,j)
                sample_idx = edge_sample_idx[node_str]
                edge_idx = self._edge2index[node_str]
                if weights is None:
                    op_weight = ms.ops.softmax(self._candidate_alphas[edge_idx][ms.Tensor.from_numpy(sample_idx).int()], axis=-1)
                else:
                    op_weight = weights[edge_idx]
                
                inp = nodes[j][0]; kwargs['st'] = nodes[j][1]
                tt_out, st_out = self._mixed_ops[self._mixed_ops_ids[node_str]](inp, x_mark, attn_mask, adj_mats, op_weight, **kwargs)
                inter_nodes_tt.append(tt_out)
                inter_nodes_st.append(st_out)
            nodes.append([sum(inter_nodes_tt), sum(inter_nodes_st)])
        # print('nodes', len(nodes))
        node_out = kwargs['node_out']
        ret_tt = 0.; ret_st = 0.
        for n,node in enumerate(nodes):
            if node_out=='other' and n==0:  # do not need nodes[0]
                continue
            # print('add node', n, node[0].shape)
            ret_tt = ret_tt + node[0]
            ret_st = ret_st + node[1]
        return ret_tt, ret_st
        # return self.norm_layer1(ret_tt), self.norm_layer2(ret_st)

    def __repr__(self):
        edge_cnt = 0
        out_str = []
        for e_id in range(self._num_edges):
            node_str = self._index2edge[e_id]

            probs = ms.ops.softmax(self._candidate_alphas[e_id], axis=0) # [num_ops, ]
            projs = self._project_weights[e_id] # [num_ops, ]
            
            # op_str = ['op:{}, prob:{:.3f}, proj:{}, info:{}'.format(i, prob, projs[i], self._mixed_ops[node_str]._candidate_ops[i]) for i,prob in enumerate(probs)]
            op_str = []
            for i, prob in enumerate(probs):
                a = "op:" + str(i)+",prob:"+str(prob)+",proj:"+str(projs[i])+",info:"+str(self._mixed_ops[self._mixed_ops_ids[node_str]]._candidate_ops[i])
                op_str.append(a)
            op_str = ',\n'.join(op_str)
            
            candidate_flag = self._candidate_flags[e_id] 
            out_str += ['mixed_op: {} {} candidate:{}\n{}\n{}'.format(e_id, node_str, candidate_flag,
                        self._mixed_ops[self._mixed_ops_ids[node_str]], op_str)]
            

        out_str += ['candidate_flag: '+','.join(['{} {}'.format(e_id, self._candidate_flags[e_id]) for e_id in range(self._num_edges)])]
        out_str += ['proj_weights: '+';'.join(['e_id {}: '.format(e_id)+','.join(['{}'.format(p_w) for p_w in self._project_weights[e_id]]) for e_id in range(self._num_edges)])]

        from ...utils.helper import add_indent
        out_str = 'STCell {{\n{}\n}}'.format(add_indent('\n'.join(out_str), 4))
        return out_str

