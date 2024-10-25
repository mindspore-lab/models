import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore.common.initializer import initializer
import numpy as npy
import scipy.sparse as sp

def calculate_normalized_laplacian(adj):
    """
    A = A + I
    L = D^-1/2 A D^-1/2
    Args:
        adj: adj matrix
    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d = npy.array(adj.sum(1))
    d_inv_sqrt = npy.power(d, -0.5).flatten()
    d_inv_sqrt[npy.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

class GCONV(nn.Cell):
    def __init__(self, num_nodes, max_diffusion_step, supports, input_dim, hid_dim, output_dim, bias_start=0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        
        self.weight = ms.Parameter(np.randn(shape))
        
        self.biases = ms.Parameter(np.randn(self._output_dim))
        
        self.unsqueeze=ops.ExpandDims()
        
        
    @staticmethod
    def _concat(x, x_):
        x_ = x_.expand_dims(0)
        return ops.concat([x, x_], 0)

    def construct(self, inputs,state):
        # 对X(t)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim)
        batch_size = inputs.shape[0]
        inputs = ops.Reshape()(inputs, (batch_size, self._num_nodes, -1))
        state = ops.Reshape()(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = ops.Concat(axis=2)([inputs, state])
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.transpose(1, 2, 0)  # (num_nodes, input_dim, batch_size)
        x0 = x0.reshape([self._num_nodes, input_size * batch_size])
        x = self.unsqueeze(x0, 0)  # (1, num_nodes, input_dim * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = np.matmul(support.to_dense(), x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * np.matmul(support.to_dense(), x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, input_dim  * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = x.reshape(self._num_matrices, self._num_nodes, input_size, batch_size)
        x = x.transpose(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = x.reshape(batch_size*self._num_nodes, input_size * self._num_matrices)
        

        x = np.matmul(x, self.weight)  # (batch_size * self._num_nodes, self._output_dim)
        x += self.biases
        # Reshape res back to 2D: (batch_size * num_node, state_dim) -> (batch_size, num_node * state_dim)
        return x.reshape(batch_size, self._num_nodes * self._output_dim)

   


# k阶扩散卷积
class DCNN(nn.Cell):
    def __init__(self,adj_mx,input_dim=1, num_units=10 ,max_diffusion_step=2, num_nodes=10, nonlinearity='tanh',
                 filter_type="laplacian"):
        """
        Args:
            input_dim: 输入维度
            num_units: 输出维度
            adj_mx: 邻接矩阵
            max_diffusion_step: 扩散步数
            num_nodes: 节点个数
            nonlinearity: 激活函数
            filter_type: "laplacian", "random_walk", "dual_random_walk"
        """

        super().__init__()
        self._activation = ops.tanh if nonlinearity == 'tanh' else ops.relu
        self.input_size = input_dim
        self.num_nodes = num_nodes
        self.output_size = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []

        supports = []

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._gconv = GCONV(self.num_nodes, self._max_diffusion_step, self._supports,
                            input_dim=input_dim, hid_dim=self.output_size, output_dim=self.output_size, bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap):
        lap = lap.tocoo()
        indices = npy.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal sparse.sparse_reorder(L)
        indices = indices[npy.lexsort((indices[:, 0], indices[:, 1]))]
        lap = ms.COOTensor(ms.Tensor(indices), ms.Tensor(lap.data), lap.shape)
        return lap

    def construct(self, inputs,state):
        return self._activation(self._gconv(inputs,state))
    
if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    adj=npy.random.randint(0,2,[10,10])
    input = np.randn([32,10,1])
    state = np.randn([32,10,10])
    model=DCNN(adj_mx=adj)
    output=model(input,state)
    print(output.shape)