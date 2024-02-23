import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
import numpy as np
import copy
import pickle
import types

def register_forward(task_num,embed_dim, field_dims):
    cast = ops.Cast()
    def snip_forward_embedding(self,ids):
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return [output * ops.expand_dims(self.weight_mask[i],-1) for i in range(task_num)] + [output*ops.expand_dims(cast(sum(self.weight_mask)==task_num,ms.int64),-1)]
    return snip_forward_embedding

# def weight_reset(m):
#     reset_parameters = getattr(m, "reset_parameters", None)
#     if callable(reset_parameters):
#         m.reset_parameters()

def sum_gd(gd, field_dims):
    offset = np.array((0, *np.cumsum(field_dims)), dtype=np.long)
    zeros = ops.Zeros()
    sum = ops.ReduceSum()
    gd_sum = zeros((len(field_dims),gd.shape[-1]),gd.dtype)
    for i in range(len(offset)-1):
        gd_sum[i,:] = sum(gd[offset[i]:offset[i+1],:],0)
    return sum(gd_sum,-1)


def mask_expand(mask, embed_dim, field_dims):
    masks = []
    for i in range(len(field_dims)):
        num =field_dims[i]
        masks.append(ops.tile(mask[i],(num,1)))
    return ops.broadcast_to(ops.concat(masks,0),(-1,embed_dim))

def return_masked_model(pruner, masks):
    cast = ops.Cast()
    model = copy.deepcopy(pruner.prun_model)
    for i in range(len(masks)):
        if i == 0:
            mask = masks[i]
            print(mask.shape)
        else: 
            mask = cast((masks[i]+mask)>0,ms.int64)
    layers = list(filter(lambda l: type(l).__name__ in pruner.forward_mapping_dict, model.emb.cells()))
    def apply_masking(mask):
        def hook(weight):
            return weight * mask
        return hook
    # for layer, _ in zip(layers, masks):
    #     mask_ = mask_expand(mask,model.emb.embed_dim,model.emb.field_dims)
    #     layer.embedding_table.data = layer.embedding_table.data * mask_
        #layer.embedding_table.register_hook(apply_masking(mask_))
    for layer in model.emb.cells():
        if type(layer).__name__ in pruner.forward_mapping_dict:
            layer.weight_mask = ops.concat([ops.expand_dims(m,0) for m in masks],0)
            layer.construct = types.MethodType(pruner.forward_mapping_dict[type(layer).__name__], layer)
    return model 

class Prunner:
    def __init__(self, model, criterion, dataloader):
        self.model = copy.deepcopy(model)
        self.field_dims = self.model.emb.field_dims
        self.task_num = model.task_num
        self.embed_dim =  self.model.emb.embed_dim
        self.prun_model = copy.deepcopy(model)#.apply(weight_reset)
        self.criterion = criterion
        self.dataloader = dataloader
        self.forward_mapping_dict = {
            'Embedding': register_forward(self.task_num,self.embed_dim, self.field_dims)
            }
        
        self.variance_scaling_init()
        self.update_forward_pass()

    def prun(self, compression_factor=0.5, num_batch_sampling=5,l=None):
        print(num_batch_sampling)
        grads_list = self.compute_grads(num_batch_sampling)
        print(grads_list)
        masks = []
        idx = []
        cast = ops.Cast()
        for i in range(self.model.task_num): #task_num
            gl = grads_list[i]
            grads = ops.flatten(gl)
            keep_params =  int((1 - compression_factor) * len(grads)) if compression_factor>0 else l[i]#
            values, idxs = ops.top_k(ops.squeeze(grads / grads.sum()), keep_params, sorted=True)
            threshold = values[-1]
            res = cast(gl / grads.sum() > threshold,ms.float32)
            print(res.shape)
            print(res.sum())
            print(idxs.asnumpy().tolist())
            idx.append(idxs.asnumpy().tolist())
            masks.append(res)
        return return_masked_model(self,masks), masks, idx

    def compute_grads(self, num_batch_sampling=1):
        grad_fn = ops.grad(self.forward_fn,None,self.model.emb.embedding.weight_mask,has_aux=True)
        moving_average_grad_list = [[]] * self.model.task_num
        for i, (c_data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            grads_list = []
            for j in range(self.model.task_num):
                grad, _ = grad_fn(c_data,labels,j)
                grads_list.append(ops.expand_dims(grad[0].abs(),0))
                      #  grads_list.append(sum_gd(torch.abs(layer.weight_mask.grad),self.model.embedding.field_dims)[j].unsqueeze(0))
            grads_list = ops.concat(grads_list,0)
            if i == 0:
                moving_average_grad_list = grads_list
            else:
                moving_average_grad_list =  ((moving_average_grad_list * i) + grads_list) / (i + 1)
            return moving_average_grad_list
    
    def forward_fn(self,data,labels,i):
        y = self.model(data)
        loss = self.criterion(y[i], labels[:, i]) 
        return loss, y

    def variance_scaling_init(self):
        for layer in self.model.emb.cells():
            if type(layer).__name__ in self.forward_mapping_dict:
                layer.weight_mask = ms.Parameter(ops.ones((self.model.task_num,len(self.field_dims)),layer.embedding_table.dtype))
                #nn.init.xavier_normal_(layer.weight)
                layer.embedding_table.requires_grad = False
    
    def update_forward_pass(self):
        for layer in self.model.emb.cells():
            if type(layer).__name__ in self.forward_mapping_dict:
                layer.construct = types.MethodType(self.forward_mapping_dict[type(layer).__name__], layer)
