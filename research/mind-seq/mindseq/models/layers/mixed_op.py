import numpy as np
import mindspore as ms
from .candidate_op import BasicOp, create_op
from .mode import Mode


class MixedOp(BasicOp): # Edge
    def __init__(self, candidate_op_profiles):
        super(MixedOp, self).__init__()
        
        self._num_ops = len(candidate_op_profiles)
        self._candidate_op_profiles = candidate_op_profiles
        self._candidate_ops = ms.nn.CellList()

        for (op_name, profile) in self._candidate_op_profiles:
            self._candidate_ops += [create_op(op_name, profile)]
    
    def ld(self, name):
        return ms.Tensor(np.load("../../Code_ALLOT/src/" + name + ".npy"), ms.float32)
    
    def construct(self, x, x_mark, attn_mask=None, adj_mats=None, weights=None, **kwargs):
        probs = weights
        t_out = 0.; s_out = 0.
        for i, idx in enumerate(self._sample_idx):
            if adj_mats is not None: # [N,N,r_graphs+num_ops]
                # idx_count = len(self._candidate_op_profiles)
                if self._candidate_op_profiles[idx][0] in ['Identity', 'Zero']:
                    op_adj_mats = None
                else:
                    r_graphs = kwargs['r_graphs']
                    op_adj_mats = ms.ops.cat([adj_mats[:,:,:r_graphs], adj_mats[:,:,r_graphs+int(idx)].unsqueeze(-1)], axis=-1)
            out, st = self._candidate_ops[int(idx)](x, x_mark, attn_mask, op_adj_mats, **kwargs)
            t_out += probs[i] * out
            s_out += probs[i] * st

        return t_out, s_out

    def set_mode(self, mode, sample_idx):
        self._mode = mode
        self._sample_idx = sample_idx

    def weight_parameters(self):
        for i in range(self._num_ops):
            for name, p in self._candidate_ops[i].parameters_and_names():
                yield p

    def num_weight_parameters(self):
        from ...utils.helper import num_parameters
        counter = 0
        for idx in self._sample_idx:
            counter += num_parameters(self._candidate_ops[int(idx)])
        return counter

    def __repr__(self):
        # mode info
        out_str = ''
        out_str += 'mode: ' + str(self._mode) + str(self._sample_idx)
        
        from ...utils.helper import add_indent
        out_str = 'mixed_op {{\n{}\n}}'.format(add_indent(out_str, 4))
        return out_str

