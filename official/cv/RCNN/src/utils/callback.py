import math
import time
import mindspore as ms
from . import logger
from .common import save_checkpoint
from .eval_utils import run_eval


class RCNNCallback(ms.Callback):
    def __init__(self, cfg, network, optimizer, eval_dataset=None):
        super(RCNNCallback, self).__init__()
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.epoch_start = time.time()
        if self.cfg.run_eval:
            self.eval_dataset = eval_dataset
        if self.cfg.run_profilor:
            self.profiler = ms.Profiler(start_profile=False)

    def on_train_begin(self, run_context):
        logger.info("Start Training")

    def on_train_epoch_begin(self, run_context):
        self.epoch_start = time.time()
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if self.cfg.run_profilor and epoch_num == 3:
            self.profiler.start()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = math.ceil(cb_params.get("cur_epoch_num", 1) / self.cfg.print_pre_epoch) + self.cfg.start_epoch
        cur_step = cb_params.get("cur_epoch_num", 1) * self.cfg.log_interval
        loss, loss_rpn, loss_rcnn = cb_params.net_outputs
        logger.info(
            f"Epoch {cur_epoch_num}/{self.cfg.epochs}, "
            f"step {cur_step % self.cfg.steps_per_epoch}/{self.cfg.steps_per_epoch}, "
            f"loss: {loss.asnumpy():.4f}, "
            f"loss_rpn: {loss_rpn.asnumpy():.4f}, "
            f"loss_rcnn: {loss_rcnn.asnumpy():.4f}, "
            f"cost {((time.time() - self.epoch_start) / cb_params.batch_num) * 1000:.2f} ms"
        )
        epoch_num = cb_params.cur_epoch_num
        if self.cfg.run_profilor and epoch_num == 5:
            self.profiler.stop()
            self.profiler.analyse()
            run_context.request_stop()
        if cb_params.get("cur_epoch_num", 1) % self.cfg.print_pre_epoch == 0:
            if self.cfg.rank == 0:
                save_checkpoint(self.cfg, cb_params.train_network, cur_epoch_num)
            if self.cfg.run_eval and cur_epoch_num > self.cfg.epochs // 2:
                run_eval(self.cfg, self.network, self.eval_dataset, cur_epoch_num, cb_params.batch_num)
