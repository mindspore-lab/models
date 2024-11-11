import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
# from mindspore.nn.transformer import OpParallelConfig
from mindspore import context, Tensor
from mindspore import Parameter, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

#context.set_context(mode=context.PYNATIVE_MODE)

class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=19, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.greater_equal = P.GreaterEqual()
        self.logical_and = P.LogicalAnd()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.logical_and(self.greater_equal(labels_int, 0), self.not_equal(labels_int, self.ignore_label))
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss

class WeightedBCEWithLogitsLoss(nn.Cell):
    
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def weighted(self, input, target, weight, alpha, beta):
        if not (ops.Size()(target) == ops.Size()(input)):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # max_val = (-input).clamp(min=0) change
        max_val = ops.clip_by_value((-input), 0, 1)

        # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() change
        exp = ops.Exp()
        loss = input - input * target + max_val + mindspore.numpy.log(exp(-max_val) + exp(-input - max_val))

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
    def construct(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


class CrossEntropy2d(nn.Cell):
    
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
    #def __init__(self, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Parameter(np.ones((class_num, 1)))
            #self.alpha = Parameter(np.ones((19, 1)))
        else:
            if isinstance(alpha, Parameter):
                self.alpha = alpha
            else:
                self.alpha = Parameter(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def construct(self, predict, target):
        #N, C, H, W = predict.size() change
        N, C, H, W = predict.shape
        sm = ops.Softmax(axis=1)
        # sm = nn.Softmax2d() change

        # P = torch.clamp(P, min = 1e-9, max = 1-(1e-9)) change
        P = sm(predict)
        P = ops.clip_by_value(P, 1e-9, 1-(1e-9))
        
        # target_mask = (target >= 0) * (target != self.ignore_label) #change
        target_mask = ops.LogicalAnd()(target >= 0, target != self.ignore_label)
        # print(target_mask.shape)

        #target = target[target_mask].view(1, -1) change
        target = ops.MaskedSelect()(target, target_mask).view(1, -1)
        # print(target.shape)


        #predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1) change
        predict_mask = ops.Tile()(target_mask.view(N, 1, H, W), (1, C, 1, 1))
        # print(predict_mask.shape)
        # print(predict.shape)
        predict = ops.MaskedSelect()(P, predict_mask).view(C, -1)


        #probs = torch.gather(predict, dim = 0, index = target) change
        dim = 0
        index = target
        # split = ops.Split(0, C)
        # predict_split = split(predict)
        # predict_split = predict_split[0]
        # print(predict_split.shape)
        # probs = ops.gather_d(predict_split, dim, index)
        probs = ops.gather_d(predict, dim, index)

        #log_p = probs.log() change
        log_p = mindspore.numpy.log(probs)


        #batch_loss = -(torch.pow((1-probs), self.gamma))*log_p  change
        pow = ops.Pow()
        batch_loss = -(pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == '__main__':

    # shape = (1, 3, 5, 5)
    # x = ops.StandardNormal(seed=2)
    # y = x(shape)
    # print(y)
    # sm = nn.Softmax(axis=1)
    # out = sm(y)
    # sm_2 = ops.Softmax(axis=1)
    # out_2 = sm_2(y)
    # equal = ops.Equal()
    # print(equal(out, out_2))

    import numpy as np

    #np.random.seed(100)

    # predict = np.random.random((1, 19, 64, 64))
    # predict = Tensor.from_numpy(predict)
    # predict = mindspore.Tensor(predict, mindspore.float32)
    # # print(predict.sum())
    #
    # target = np.random.randint(0, 1, (1, 64, 64))
    # target = Tensor.from_numpy(target)
    # # print(target.shape)
    # target = mindspore.Tensor(target, mindspore.int32)
    # target_mask = ops.LogicalAnd()(target >= 0, target != 255)
    # target = target[target_mask].view(1, -1)
    # print(target)

    #print(target.dtype, predict.dtype)

    # y = np.random.randint(0, 255, (2, 2))
    # y = Tensor.from_numpy(y)
    # print(y)
    #
    # mask = ops.LogicalAnd()(y >= 50, y != 255)
    # #mask = (y <= 50) * (y != 255)
    # print(mask)
    # y = ops.MaskedSelect()(y, mask)
    #
    # print(y)
    # x = np.array([[0.4714, 0.4963], [0.8436, 0.8586]])
    # x = Tensor.from_numpy(x)
    # print(x)
    # log = mindspore.numpy.log(x)
    # print(log)


    # criterion = CrossEntropy2d(19)
    # criterion = WeightedBCEWithLogitsLoss()
    # loss = CrossEntropy2d(19)(predict, target)
    # print(loss)

    # x = np.random.random((6))
    # x = Tensor.from_numpy(x)
    # print(x)
    # split = ops.Split(0, 2)
    # out = split(x)
    # first = out[0]
    # print(out)
    # print(first)

    x = np.random.random((2, 2))
    x = mindspore.Tensor(x)
    print(x.dtype)



