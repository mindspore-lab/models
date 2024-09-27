from mindspore import Callback
import numpy as np



class EvalCallBack(Callback):
    """
        默认validation程序，此方法监控的metrics为loss
    """
    def __init__(self, model, eval_dataloader, epochs_to_eval, per_eval, logger,optim=None):
        self.model = model
        self.eval_dataloader = eval_dataloader
        # epochs_to_eval是一个int数字，代表着：每隔多少个epoch进行一次验证
        self.epochs_to_eval = epochs_to_eval
        self.per_eval = per_eval
        self._logger=logger
        self.optim=optim

    def epoch_end(self, run_context):
        # 获取到现在的epoch数
        if self.optim!=None:
            self._logger.info("learning rate is: {}".format(self.optim.get_lr().asnumpy()))
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        # 如果达到进行验证的epoch数，则进行以下验证操作
        if cur_epoch % self.epochs_to_eval == 0:
            if hasattr(self.model, 'validate') and callable(getattr(self.model, 'validate')):
                self.model.validate()
                self._logger.info("validating...")
            # 此处model设定的metrics是准确率Accuracy
            losses = []
            for batch in self.eval_dataloader:
                loss = self.model(*batch)
                self._logger.debug(loss)
                losses.append(loss.asnumpy())
            mean_loss = np.mean(losses)
            self.per_eval["epoch"].append(cur_epoch)
            self.per_eval["loss"].append(mean_loss)
            self._logger.info("valid loss 为: {}".format(mean_loss))
            if mean_loss < self.per_eval['min_val_loss']:
                self._logger.info("Val loss decrease from {} to {}".format(self.per_eval['min_val_loss'],mean_loss))
                self.per_eval['best_epoch']=cur_epoch
                self.per_eval['min_val_loss'] = mean_loss
            if hasattr(self.model, 'validate') and callable(getattr(self.model, 'validate')):
                self.model.train()




