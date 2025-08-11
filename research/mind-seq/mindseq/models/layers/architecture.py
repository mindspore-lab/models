import mindspore as ms
import mindspore.nn as nn
class Architecture(nn.Cell):
    def __init__(self, model):
        super(Architecture, self).__init__()

        self._model = model

    def construct(self, x, x_mark, attn_mask, adj_mats, mode, weights, **kwargs):
        return self._model(x, x_mark, attn_mask, adj_mats, mode, weights, **kwargs)

    def next_cell(self, cell_id):
        for idx in range(cell_id, len(self._model._layers)):
            if isinstance(self._model._layers[idx], self._model._cell_type):
                remain_eids = ms.ops.nonzero(self._model._layers[idx]._candidate_flags).asnumpy().T[0]
                if len(remain_eids)!=0:
                    return idx
                else:
                    continue
        return len(self._model._layers)
    
    def remain_edge_ids(self, cell_id):
        return ms.ops.nonzero(self._model._layers[cell_id]._candidate_flags).cpu().numpy().T[0]

    def project_masked_weights(self, cell_id, e_id, op_id):
        num_cells = self._model.num_cells()
        weights_total = [None] * num_cells
            
        weights = self._model._layers[cell_id].project_parameters()
        proj_mask = ms.ops.ones_like(weights[e_id])
        proj_mask[op_id] = 0
        weights[e_id] = weights[e_id] * proj_mask
        weights_total[cell_id] = weights

        return weights_total

    def project_op(self, cell_id, e_id, op_id):
        self._model._layers[cell_id].project_op(e_id, op_id)

    def set_mode(self, mode):
        self._model.set_mode(mode)

    def arch_parameters(self):
        return self._model.arch_parameters()
    
    def proj_parameters(self):
        return self._model.proj_parameters()

    def weight_parameters(self):
        return self._model.weight_parameters()
    
    def num_weight_parameters(self):
        return self._model.num_weight_parameters()

    def num_cells(self):
        return self._model.num_cells()

    def num_layers(self):
        return len(self._model._layers)

    def num_ops(self, cell_id=None):
        return self._model.num_ops(cell_id)
    
    def num_edges(self, cell_id=None):
        return self._model.num_edges(cell_id)

    def __repr__(self):
        out_str = [str(self._model)]

        from ...utils.helper import add_indent
        out_str = 'Architecture {{\n{}\n}}\n'.format(add_indent('\n'.join(out_str), 4))
        
        return out_str


