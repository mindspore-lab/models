from __future__ import absolute_import

import numpy as np

from mindspore import _checkparam as Validator
from mindspore.train.callback._callback import _handle_loss
from mindnlp.abc import Callback
from mindspore._c_expression import _collect_host_info


class LossMonitor(Callback):
    """
    Monitor the loss in train or monitor the loss and eval metrics in fit.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: ``1`` .

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.train import Model, LossMonitor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> loss_monitor = LossMonitor()
        >>> model.train(10, dataset, callbacks=loss_monitor)
    """

    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        Validator.check_non_negative_int(per_print_times)
        self._per_print_times = per_print_times
        self._last_print_time = 0

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        _collect_host_info("Callback", "LossMonitor", "step_end", level=1)
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        loss = _handle_loss(cb_params.net_outputs)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("In epoch: {} step: {}, loss is NAN or INF, training process cannot continue, "
                             "terminating training.".format(cur_epoch_num, cur_step_in_epoch))

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" % (cur_epoch_num, cur_step_in_epoch, loss), flush=True)

    def on_train_epoch_end(self, run_context):
        """
        When LossMonitor used in `model.fit`, print eval metrics at the end of epoch if current epoch
        should do evaluation.

        Args:
            run_context (RunContext): Include some information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        _collect_host_info("Callback", "LossMonitor", "train_epoch_end", level=1)
        metrics = cb_params.get("metrics")
        if metrics:
            print("Eval result: epoch %d, metrics: %s" % (cb_params.cur_epoch_num, metrics))

