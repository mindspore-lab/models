import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops 
import numpy as npy

class GCNLayer(nn.Cell):
    def __init__(self, num_of_features, num_of_filter):
        """
        One layer of GCN

        Arguments:
        num_of_features {int} -- the dimension of node feature
        num_of_filter {int} -- the number of graph filters
        """
        super(GCNLayer, self).__init__()
        self.gcn_layer = nn.SequentialCell(
            nn.Dense(in_channels=num_of_features,
                     out_channels=num_of_filter),
            nn.ReLU()
        )

    def construct(self, input_, adj):
        """
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)

        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size, _, _ = input_.shape
        adj = np.tile(adj, (batch_size, 1, 1))
        batmatmul = ops.BatchMatMul()
        input_ = batmatmul(adj, input_)
        output = self.gcn_layer(input_)
        return output
    
class GCLSTM(nn.Cell):
    def __init__(self, adj,input_size=1,hidden_size=10,bias=True):
        super(GCLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gcn = GCNLayer(input_size,input_size)
        self.x2h = nn.Dense(self.input_size, 4 * self.hidden_size)
        self.h2h = nn.Dense(self.hidden_size, 4 * self.hidden_size)
        self.adj = adj
        self.sigmoid=ops.Sigmoid()



    def construct(self, x, hidden):
        """
        f/i/o = sigmoid( W(f(x)) + U(hidden) +bias)
        c = tanh( W(f(x)) + U(hidden) +bias)
        C_hot = f * context + i * c
        h_hot = o * tanh(c_hot)
        Args:
            x: shape (batch,num_nodes,input_dim)
            hidden: shape (2,batch,num_nodes,hidden_size)
        Returns:
            tuple:(hy tensor,
                    cy tensor)
        """
        hx, cx = hidden
        x = self.gcn(x,self.adj)

        gates = self.x2h(x) + self.h2h(hx)  # (batch*num_nodes,4*hidden)

        ingate, forgetgate, cellgate, outgate = ops.split(gates, -1, 4)  # (batch*num_nodes,hidden)

        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        outgate = self.sigmoid(outgate)
        cellgate = ops.tanh(cellgate)

        cy = cx*forgetgate + ingate*cellgate

        hy = outgate*ops.tanh(cy)
        return (hy, cy)
    
if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE,device_target='CPU')
    dim=1
    adj=np.randn([10,10])
    input = np.randn([32,10,1])
    state = np.randn([2,32,10,10])
    model=GCLSTM(adj=adj)
    output,_=model(input,state)
    print(output.shape)