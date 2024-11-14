import os
import time
import numpy as np
import mindspore as ms
from . import logger
from .common import save_checkpoint
from .metrics import GetConfusionMatrix


class Callback(ms.Callback):
    """
    Monitor the loss in train and eval metrics in eval.
    Args:
        cfg (obj): configs
        network (Cell): Train network.
        optimizer (Cell): Train optimizer
        eval_net (Cell): Evaluation network.
        eval_dataset (obj): Evaluation dataset
    """
    def __init__(self, cfg, network, optimizer, eval_net=None, eval_dataset=None):
        super(Callback, self).__init__()
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.epoch_start = time.time()
        self.eval_dataset = eval_dataset
        self.eval_net = eval_net
        self.get_confusion_matrix = GetConfusionMatrix(self.cfg.num_classes,
                                                       self.cfg.data.ignore_label,
                                                       1)
        if self.cfg.run_profilor:
            self.profiler = ms.Profiler(start_profile=False)
        if self.cfg.run_eval:
            self.best_metric = 0

    def run_eval(self):
        self.eval_net.set_train(False)
        num_classes = self.cfg.num_classes
        data_loader = self.eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
        confusion_matrix = np.zeros((num_classes, num_classes))
        item_count = 0
        for data in data_loader:
            img = ms.Tensor(data["image"])
            label = np.squeeze(data["label"])
            pred = self.eval_net(img)
            pred = np.squeeze(pred.asnumpy()).astype(np.uint8)
            label = label.astype(np.uint8)
            confusion_matrix += self.get_confusion_matrix(label, pred)
            item_count += 1
        logger.info(f"Total number of images: {item_count}")

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        iou_array = tp / np.maximum(1.0, pos + res - tp)
        mean_iou = iou_array.mean()
        if mean_iou >= self.best_metric and self.cfg.rank == 0:
            self.best_metric = mean_iou
            logger.info(f"update best miou {mean_iou}")
            ms.save_checkpoint(self.eval_net,
                               os.path.join(self.cfg.save_dir, "checkpoints",
                                            f"{self.cfg.net}_{self.cfg.backbone.initializer}_miou_{mean_iou}.ckpt")
                               )
        self.eval_net.set_train(True)

        # Show results
        logger.info(f"=========== Evaluation Result ===========")
        logger.info(f"iou array: \n {iou_array}")
        logger.info(f"miou: {mean_iou}")

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
        cur_step = cb_params.get("cur_epoch_num", 1) * self.cfg.log_interval + self.cfg.start_step
        loss, cond, scaling_sens = cb_params.net_outputs
        step = self.optimizer.global_step
        if self.optimizer.dynamic_lr:
            cur_lr = self.optimizer.learning_rate(step - 1)[0].asnumpy()
        else:
            cur_lr = self.optimizer.learning_rate.asnumpy()
        logger.info(
            f"step {cur_step}/{self.cfg.total_step}, "
            f"loss: {loss.asnumpy():.4f}, "
            f"cur_lr: {cur_lr:.4f}, "
            f"cost {((time.time() - self.epoch_start) / cb_params.batch_num) * 1000:.2f} ms"
        )
        epoch_num = cb_params.cur_epoch_num
        if self.cfg.run_profilor and epoch_num == 5:
            self.profiler.stop()
            self.profiler.analyse()
            run_context.request_stop()
        if cur_step % 5000 == 0 and cur_step != 0:
            save_checkpoint(self.cfg, self.network, None, cur_step)
            if self.cfg.run_eval and cur_step > self.cfg.total_step // 2:
                try:
                    self.run_eval()
                except:
                    import traceback

                    traceback.print_exc()
