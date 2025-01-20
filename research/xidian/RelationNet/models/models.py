# Copyright 2024 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.ops as ops
from mindspore import nn
from mindspore.common.initializer import Normal, initializer
from mindspore.ops import functional as F




class MyWithLossCell(nn.Cell):
   def __init__(self, Net1,Net2,loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=True)
       self.net1 = Net1
       self.net2 = Net2
       self._loss_fn = loss_fn
       
   def construct(self, samples, batches, label):
       sample_features = self.net1(samples) # 5x64*5*5
       batch_features = self.net1(batches) # 20x64*5*5
       
       ## 制作标签
       sample_features_ext = sample_features.unsqueeze(0).tile((95, 1, 1, 1, 1))
       batch_features_ext = batch_features.unsqueeze(0).tile((5, 1, 1, 1, 1))
       batch_features_ext = ops.swapaxes(batch_features_ext, 0, 1)
       
       ### Bak
       relation_pairs = ops.cat((sample_features_ext, batch_features_ext), 2).view(-1, 128, 5, 5)  
       relations = self.net2(relation_pairs).view(-1,5)            
       loss = self._loss_fn(relations,label)
       return loss





class TrainOneStepCellV2(nn.TrainOneStepCell):
    '''Build train network.'''
    def __init__(self, network, optimizer, sens=None, return_grad=False):
        super(TrainOneStepCellV2, self).__init__(network, optimizer, sens=1.0, return_grad=False)
        self.GRADIENT_CLIP_TYPE = 1
        self.GRADIENT_CLIP_VALUE = 0.5
        self.clip_gradients = ClipGradients()
    
    def construct(self, *inputs):
        if not self.sense_flag:
            return self._no_sens_impl(*inputs)
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        grads = self.clip_gradients(grads, self.GRADIENT_CLIP_TYPE, self.GRADIENT_CLIP_VALUE)
        loss = F.depend(loss, self.optimizer(grads))
        if self.return_grad:
            grad_with_param_name = {}
            for index, value in enumerate(grads):
                grad_with_param_name[self.weights_name[index]] = value
            return loss, grad_with_param_name
        return loss


class ClipGradients(nn.Cell):
    """
    Clip gradients.
    Inputs:
        grads (tuple[Tensor]): Gradients.
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
    Outputs:
        tuple[Tensor], clipped gradients.
    """
 
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
 
    def construct(self, grads, clip_type, clip_value):
        if clip_type != 0 and clip_type != 1:
            return grads
 
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = ops.clip_by_value(grad, self.cast(F.tuple_to_array((clip_value,)),dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)),dt))
                new_grads = new_grads + (t,)

        return new_grads
