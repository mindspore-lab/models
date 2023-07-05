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
"""Self-defined callbacks."""
from scipy import interpolate

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
            fpr95 = self.inference()
            print(f'Test FPR95: {fpr95}', flush=True)

    def inference(self):
        self.metric.clear()
        for data in self.eval_dataset.create_dict_iterator():
            pred_scores = self.model(data["image1"], data["image2"])
            self.metric.update(pred_scores, data["matches"])
        fpr, tpr, thresholds = self.metric.eval()
        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        return fpr95