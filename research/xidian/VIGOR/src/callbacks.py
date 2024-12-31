# Copyright 2024 Xidian University
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
import os
import time
import numpy as np

from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from mindspore import Tensor


class RecallEvalCallback(Callback):
    """Callback for inference while training. Dataset cityscapes."""
    def __init__(self, query_loader, ref_loader, model,
                 start_epoch=0, save_path=None, interval=1, 
                 query_data_size=53694, ref_data_size=46563):
        super(RecallEvalCallback, self).__init__()
        self.query_loader = query_loader
        self.ref_loader = ref_loader
        self.model = model

        self.query_data_size = query_data_size
        self.ref_data_size = ref_data_size
        
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.interval = interval
        self.best_rAt1 = 0

    def on_train_epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0:
                self.model.set_train(False)
                
                device_id = cb_param.device_number
                # device_id = int(os.getenv("DEVICE_ID"))
                
                RAt1, RAt5, RAt10 = self.inference()
                if RAt1 > self.best_rAt1:
                    self.rAt1 = RAt1
                    if self.save_path:
                        file_path = os.path.join(self.save_path, f"best-{device_id}.ckpt")
                        save_checkpoint(self.model, file_path)
                self.model.set_train(True)

    def inference(self):
        query_features, reference_features, query_labels = self.get_descriptor()
        RAt1, RAt5, RAt10 = self.validate(query_features, reference_features, query_labels)
        return RAt1, RAt5, RAt10
    
    def get_descriptor(self):
        query_features = np.zeros([self.query_data_size, 4096])
        query_labels = np.zeros([self.query_data_size], dtype=np.int32)
        reference_features = np.zeros([self.ref_data_size, 4096])

        for batch in self.query_loader:
            image, index, label = batch
            query_embed = self.model.inference(image, 'query')
            query_features[index.asnumpy(), :] = query_embed.asnumpy()
            query_labels[index.asnumpy()] = label.asnumpy()

        for batch in self.ref_loader:
            image, index = batch
            reference_embed = self.model.inference(image, 'ref')
            reference_features[index.asnumpy().astype(int), :] = reference_embed.asnumpy()

        return query_features, reference_features, query_labels
    
    def validate(self, query_features, reference_features, query_labels, topk=[1, 5, 10]):
        ts = time.time()
        N = query_features.shape[0]
        M = reference_features.shape[0]
        topk.append(M//100)
        results = np.zeros([len(topk)])
        if N < 80000:
            query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

            for i in range(N):
                ranking = np.sum((similarity[i,:]>similarity[i, query_labels[i]])*1.)

                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.
        else:
            # split the queries if the matrix is too large, e.g. VIGOR
            assert N % 4 == 0
            N_4 = N // 4
            for split in range(4):
                query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
                query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
                query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
                reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
                similarity = np.matmul(query_features_i / query_features_norm,
                                    (reference_features / reference_features_norm).transpose())
                for i in range(query_features_i.shape[0]):
                    ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                    for j, k in enumerate(topk):
                        if ranking < k:
                            results[j] += 1.

        results = results/ query_features.shape[0] * 100.
        print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
        return results[:3]


class TimeLossMonitor(Callback):
    def __init__(self, lr_init=None):
        super(TimeLossMonitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
        self.losses = []
        self.epoch_time = 0
        self.step_time = 0

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch: [{:3d}/{:3d}], epoch time: {:5.3f}, steps: {:5d}, "
              "per step time: {:5.3f}, avg loss: {:5.3f}, lr:[{:8.6f}]".format(
                  cb_params.cur_epoch_num, cb_params.epoch_num, epoch_mseconds, cb_params.batch_num,
                  per_step_mseconds, np.mean(self.losses), self.lr_init[cb_params.cur_step_num - 1]), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
