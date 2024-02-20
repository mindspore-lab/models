import mindspore
import numpy as np
from mindspore import Tensor, ops, nn
from mindspore import context
# self.dot = ops.einsum('ac,bc->abc', self.u_g_embeddings, self.pos_i_g_embeddings)
# self.batch_ratings = ops.einsum('ajk,lk->aj', self.dot, self.r2)    
context.set_context(device_target='GPU',device_id=0)
x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
leaky_relu = nn.LeakyReLU()
output = leaky_relu(x)
print(output)