import time

from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback
from models.dlinknet import DLinkNet34, DLinkNet50


class MyCallback(Callback):
    def __init__(self, weight_file_name, rank_label, device_num, show_step=False, lr=2e-4, model_name='dlinknet34'):
        super(MyCallback, self).__init__()
        self.no_optim = 0
        self.train_epoch_best_loss = 100.
        self.tic_begin = time.time()
        self.file_name = weight_file_name
        self.rank_label = rank_label
        self.device_num = device_num
        self.show_step = show_step
        self.lr = lr
        self.model_name = model_name
        self.current_epoch_loss_sum = 0
        self.step_count = 0
        self.step_per_epoch = 0

    def begin(self, run_context):
        """Called once before the network executing."""

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        print(self.rank_label + time.strftime("%Y-%m-%d %X", time.localtime()) + ' epoch ' + str(epoch_num) + ' start!')

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        train_step_loss = cb_params.net_outputs
        cur_time = str(int(time.time() - self.tic_begin))
        optimizer = cb_params.optimizer
        print(self.rank_label + "epoch_end {} step {}, loss is {}, scale sense is {}, overflow is {}".format(
            epoch_num, step_num, train_step_loss[0].asnumpy(), train_step_loss[2].asnumpy(),
            train_step_loss[1].asnumpy()))
        # compute train_epoch_loss
        train_epoch_loss = self.current_epoch_loss_sum / self.step_count
        if self.step_per_epoch == 0:
            self.step_per_epoch = self.step_count
        self.current_epoch_loss_sum = 0
        self.step_count = 0

        print(self.rank_label + 'epoch:' + str(epoch_num) + '    time:' + cur_time)
        print(self.rank_label + 'train_loss:' + str(train_epoch_loss))

        if train_epoch_loss >= self.train_epoch_best_loss:
            self.no_optim += 1
        else:
            self.no_optim = 0
            self.train_epoch_best_loss = train_epoch_loss
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=self.file_name)
            print(self.rank_label + "Save the minimum train loss checkpoint, the loss is", train_epoch_loss)

        if self.no_optim > 6 and self.device_num == 1:
            print(self.rank_label + 'early stop at %d epoch (cause no_optim > 6)' % epoch_num)
            run_context.request_stop()
            return
        if self.no_optim > 3 * max(1, self.device_num / 2):
            if self.lr < 5e-7 and self.device_num == 1:
                print(self.rank_label + 'early stop at %d epoch (cause cur_lr < 5e-7)' % epoch_num)
                run_context.request_stop()
                return
            if self.model_name == 'dlinknet34':
                network = DLinkNet34()
            else:
                network = DLinkNet50()
            param_dict = load_checkpoint(self.file_name)
            load_param_into_net(network, param_dict)
            cb_params.train_network = network
            old_lr = self.lr
            self.lr = self.lr / 5.0
            optimizer.learning_rate.set_data(self.lr)
            print(self.rank_label + 'update learning rate: ' + str(old_lr) + ' -> ' + str(self.lr))
            # reset no_optim to 0
            self.no_optim = 0

    def step_begin(self, run_context):
        """Called before each step beginning."""
        pass

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        if self.step_per_epoch != 0:
            step_num %= self.step_per_epoch
        train_step_loss = cb_params.net_outputs
        if self.show_step:
            print(self.rank_label + "epoch {} step {}, loss is {}, scale sense is {}, overflow is {}".format(
                epoch_num, step_num, train_step_loss[0].asnumpy(), train_step_loss[2].asnumpy(),
                train_step_loss[1].asnumpy()))
        self.current_epoch_loss_sum += train_step_loss[0].asnumpy()
        self.step_count += 1

    def end(self, run_context):
        """Called once after network training."""
        pass
