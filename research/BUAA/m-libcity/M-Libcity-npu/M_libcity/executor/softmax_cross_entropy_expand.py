from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore
import numpy as np
from mindspore import ops
import model.loss

from mindspore import log as logger
 
class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False , logger=None):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        self.loss = nn.MSELoss(reduction = 'mean')
        self._logger = logger
 
    def construct(self, logit, label):
        #logit = ops.tuple_to_array(logit)
        #label = ops.tuple_to_array(label)
        logfile=open('file.log','w')
        
        length = len(logit)
        logfile.write("***********\n")
        logfile.write(str(length))
        logfile.write(str(label[0]))
        loss = Tensor(0.0, mstype.float32)
        for i in range(length):
            temp = label[i, 0, :, 0]
            loss += (self.loss(logit[i], temp))
        loss = loss / length

        return loss
    
if __name__ == '__main__':
    SoftmaxCrossEntropyExpand(sparse = False)