import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import initializer as init
import mindspore.ops as ops
from tqdm import tqdm
from .DBN import DBN

class Model_NN(nn.Cell):
    """HardNet model definition
    """
    def __init__(self,in_dim=1568,init_weight=None):
        super(Model_NN, self).__init__()
        self.features = nn.SequentialCell(
            nn.Dense(in_dim, 1568,activation=nn.Sigmoid()),
            nn.Dense(1568, 784,activation=nn.Sigmoid()),
            nn.Dense(784, 100,activation=nn.Sigmoid()),
            nn.Dense(100, 1),
            nn.Sigmoid(),
        )
        # initialize by developer
        self._initialize_weights(init_weight)

    def _initialize_weights(self,weight):  #!!!
        """
        Init the weight of Conv2d and Dense in the net.
        """
        pos = 0
        if weight is not None:
            for _, cell in self.cells_and_names():

                if isinstance(cell, nn.Dense) and pos < 3:
                    cell.weight.set_data(init.initializer(Tensor(weight[pos].W,dtype=mindspore.float32),     #!!!
                                                      weight[pos].W.shape,
                                                      cell.weight.dtype))
                    cell.bias.set_data(init.initializer(Tensor(weight[pos].hbias,dtype=mindspore.float32),     #!!!
                                  weight[pos].hbias.shape,
                                  cell.bias.dtype))
                    pos = pos +1

    def construct(self, input):
        x_features = self.features(input)
        return x_features

class ModelWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(ModelWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data,label):
        out = self._backbone(data)
        return self._loss_fn(out, label)


def DBN_pretrain(args,data_loader):

    dbn = DBN( n_ins=1568, hidden_layer_sizes=[1568,784,100], n_outs=100)
    step=args.num_train//args.batchsize

    for i in range(dbn.n_layers):

        for epoch in range(args.DBN_numepochs):
            err = 0
            for j,(img,label) in enumerate(data_loader):
                input=img
                err += dbn(input,i)
            print('Average reconstruction error is:',err/step)
            with open('err.txt', "a+") as f:
                f.write('Average reconstruction error is: {:.8f}\n'.format((err/step).asnumpy()))
    return dbn.rbm_layers

def train_loop(model, dataset, loss_fn, optimizer,args):
    import time
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss
    model.set_train()
    for t in range(args.NN_numepochs):
        sum_loss = 0
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        pbar = tqdm(enumerate(dataset.create_tuple_iterator()))
        for batch, (data, label) in pbar:
            start_time = time.time()
            loss = train_step(data, label)
            end_time=time.time()
            times=(end_time-start_time)*1000
            loss= loss.asnumpy()
            sum_loss +=loss
            if batch % 10 == 0:
                pbar.set_description("epoch:[{:3d}/{:3d}], MSEloss: {:>7f}, per step time:{:5.3f} ms ".format(t+1,args.NN_numepochs,sum_loss/(batch+1),times))


