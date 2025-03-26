import mindspore as ms

class gconv(ms.nn.Cell):
    def __init__(self):
        super(gconv, self).__init__()

    def construct(self, x, A):
        x = x.swapaxes(-1, -2).matmul(A).swapaxes(-1, -2)
        return x

class GraphConv(ms.nn.Cell):
    def __init__(self, c_in, c_out, n_graphs, order, use_bn=True, dropout=0.1):
        super(GraphConv,self).__init__()
        c_in = (order * n_graphs + 1) * c_in

        self.n_graphs = n_graphs
        self.order = order
        self.use_bn = use_bn

        self.gconv = gconv()
        self.linear = ms.nn.Dense(c_in, c_out)
        self.dropout = ms.nn.Dropout(p=dropout)
        if use_bn:
            self.bn = ms.nn.BatchNorm2d(c_out)

    def construct(self, x, adj_mats, **kwargs):
        # x: [B, D, N, T]
        out = [x]

        for i in range(self.n_graphs):
            y = x
            for j in range(self.order):
                y = self.gconv(y, adj_mats[:, :, i].squeeze())
                out += [y]
        x = ms.ops.cat(out, axis=1)
        x = self.linear(x.swapaxes(1, -1)).swapaxes(1, -1)
        if self.use_bn:
            x = self.bn(x)
        x = self.dropout(x)
        return x # [B, D, N, T]

class nconv(ms.nn.Cell):
    def __init__(self):
        super(nconv, self).__init__()

    def construct(self, x, A):
        x = x.matmul(A)
        return x

class NGraphConv(ms.nn.Cell):
    def __init__(self, c_in, c_out, n_graphs, order, use_bn=True, dropout=0.1):
        super(NGraphConv,self).__init__()
        c_in = (order * n_graphs + 1) * c_in

        self.n_graphs = n_graphs
        self.order = order
        self.use_bn = use_bn

        self.nconv = nconv()
        self.linear = ms.nn.Conv1d(c_in, c_out, kernel_size=1, 
                                    padding=0, stride=1, has_bias=True)
        self.dropout = ms.nn.Dropout(p=dropout)
        if use_bn:
            self.bn = ms.nn.BatchNorm1d(c_out)

    def construct(self, x, adj_mats, **kwargs):
        # x: [B, D, N]
        out = [x]

        for i in range(self.n_graphs):
            y = x
            for j in range(self.order):
                y = self.nconv(y, adj_mats[:, :, :, i].squeeze())
                out += [y]
            
        x = ms.ops.cat(out, axis=1)
        x = self.linear(x)
        
        if self.use_bn:
            x = self.bn(x)
        x = self.dropout(x)
        return x # [B, D, N]