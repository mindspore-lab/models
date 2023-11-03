# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Load pretrained model."""

import os

from mindspore import load_checkpoint, load_param_into_net


class LoadPretrainedModel():
    """Load pretrained model from url."""

    def __init__(self, model, url):
        self.model = model
        self.url = url
        self.path = os.path.join('./', self.__class__.__name__)

    # def download_checkpoint_from_url(self):
    #     """Download the checkpoint if it doesn't exist already."""
    #     os.makedirs(self.path, exist_ok=True)

    #     # download files
    #     self.download_url(self.url, path=self.path)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.param_dict = load_checkpoint(os.path.join(self.path, os.path.basename(self.url)))

    def load_param_into_net(self):
        load_param_into_net(self.model, self.param_dict)

    def run(self):
        """Download checkpoint file and load it."""
        # self.download_checkpoint_from_url()
        self.load_checkpoint()
        self.load_param_into_net()
