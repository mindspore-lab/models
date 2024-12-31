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
from mindspore import ops as ops
from sklearn import metrics



def eval_generator(encoder, classifier, data_loader):
    encoder.set_train(False)
    classifier.set_train(False)
    acc = 0
    size = data_loader.get_dataset_size()
    for _, (data, label) in enumerate(data_loader):
        y_test_pred = classifier(encoder(data))
        predict, _ = ops.max(y_test_pred, axis=1)
        acc += metrics.accuracy_score(predict.asnumpy(), label.asnumpy())
    return acc/size
