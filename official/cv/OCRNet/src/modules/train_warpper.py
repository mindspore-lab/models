import mindspore as ms
from mindspore import ops, nn

_grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    r"""Network training with loss scaling"""

    def __init__(self, network, optimizer, scale_sense, clip_grad=False, force_update=False):
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.clip_grad = clip_grad
        self.force_update = ms.Tensor(force_update, ms.bool_)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * scaling_sens.astype(loss.dtype)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if ops.logical_or(self.force_update, not overflow):
            loss = ops.depend(loss, self.optimizer(grads))

        return loss, cond, scaling_sens
