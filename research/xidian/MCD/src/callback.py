# Copyright 2023 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from mindspore import Callback


class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, metric, eval_per_epoch):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.metric = metric

    def on_train_epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc1, acc2, acc_ensemble, test_loss = self.inference()
            print(
                f'Test set: acc1:{acc1.eval():.4f}\tacc2:{acc2.eval():.4f}\tacc_ensemble:{acc_ensemble.eval():.4f}\ttest_loss:{test_loss.asnumpy():.4f}',
                flush=True)

    def inference(self):
        self.metric['acc1'].clear()
        self.metric['acc2'].clear()
        self.metric['acc_ensemble'].clear()
        size = 0
        test_loss = 0
        for data in self.eval_dataset.create_dict_iterator():
            img = data['T']
            label = data['T_label']
            pred1, pred2, pred_ensemble, loss = self.model(img, label)
            test_loss += loss
            k = label.shape[0]
            size += k
            self.metric['acc1'].update(pred1, label)
            self.metric['acc2'].update(pred2, label)
            self.metric['acc_ensemble'].update(pred_ensemble, label)
        test_loss /= size
        return self.metric['acc1'], self.metric['acc2'], self.metric['acc_ensemble'], test_loss
