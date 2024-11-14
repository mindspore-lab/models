import mindspore as ms
import mindspore.numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as npy
import scipy.sparse as sp
from mindspore import context

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

class GCGRU(nn.Cell):
    def __init__(self, num_units=10, adj_mx=None, num_nodes=10, input_dim=1):
        # ----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.act = ops.tanh

        # 这里提前构建好拉普拉斯
        support = calculate_normalized_laplacian(adj_mx)
        self.normalized_adj = self._build_sparse_matrix(support)
        self.sigmoid=nn.Sigmoid()
        self.init_params()

    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        self.weight_0 = ms.Parameter(np.randn((input_size, 2 * self.num_units)))
        self.bias_0 = ms.Parameter(np.randn(2 * self.num_units))
        self.weight_1 = ms.Parameter(np.randn((input_size, self.num_units)))
        self.bias_1 = ms.Parameter(np.randn(self.num_units))
        

    @staticmethod
    def _build_sparse_matrix(lap):
        lap = lap.tocoo()
        indices = npy.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal sparse.sparse_reorder(L)
        indices = indices[npy.lexsort((indices[:, 0], indices[:, 1]))]
        indices = ms.Tensor(indices)
        lap = ms.COOTensor(indices, ms.Tensor(lap.data), lap.shape)
        return lap

    def construct(self, inputs, state):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)
        Returns:
            tensor: shape (B, num_nodes * gru_units)
        """

        inputs=inputs.astype(ms.float32)
        state=state.astype(ms.float32)
        output_size = 2 * self.num_units
        value = self.sigmoid(
            self._gc(inputs, state, output_size, bias_start=1.0))  # (batch_size, self.num_nodes, output_size)
        r, u = np.split(value, 2, axis=-1)
        r =r.reshape(-1, self.num_nodes * self.num_units)  # (batch_size, self.num_nodes * self.gru_units)
        u = u.reshape(-1, self.num_nodes * self.num_units)

        c = self.act(self._gc(inputs, r * state, self.num_units))
        c = c.reshape(-1, self.num_nodes * self.num_units)
        new_state = u * state + (1.0 - u) * c
        return new_state

    def _gc(self, inputs, state, output_size, bias_start=0.0):
        """
        GCN
        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            output_size:
            bias_start:
        Returns:
            tensor: (B, num_nodes , output_size)
        """
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.num_nodes, -1)  # (batch, self.num_nodes, self.dim)
        
        state = state.reshape (batch_size, self.num_nodes, -1)  # (batch, self.num_nodes, self.gru_units)
        inputs_and_state = ops.concat([inputs, state], axis=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.transpose(1, 2, 0)  # (num_nodes, dim+gru_units, batch)
        x0 = x0.reshape([self.num_nodes, -1])
        self.normalized_adj=self.normalized_adj.astype(ms.float32)
        x1 = ops.matmul(self.normalized_adj.to_dense(), x0)  # A * X

        x1 = x1.reshape(self.num_nodes, input_size, batch_size)
        x1 = x1.transpose(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1 = x1.reshape([-1, input_size])  # (batch_size * self.num_nodes, input_size)

        weights = None
        if output_size == self.num_units:
            weights = self.weight_1
        else :
            weights = self.weight_0

        x1 = ops.matmul(x1, weights)  # (batch_size * self.num_nodes, output_size)
        biases=None
        if output_size== self.num_units:
            x1 += self.bias_1
        else :
            x1 += self.bias_0
    
        x1 = x1.reshape(batch_size, self.num_nodes, output_size)
        return x1
    
if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE,device_target='CPU')
    N=20
    dim=1
    adj=npy.random.randint(0,2,[10,10])
    input = np.randn([1,10*1])
    state = np.randn([1,10*10])
    model=GCGRU(adj_mx=adj)
    output=model(input,state)
    print(output.shape)