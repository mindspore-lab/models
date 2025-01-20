# Copyright 2021 Xidian University
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
'''
 Deep Belief Nets (DBN)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

'''
from src.Utils import *
from .RBM import RBM

class DBN(nn.Cell):
    def __init__(self, \
                 n_ins=1568, hidden_layer_sizes=[1568, 784, 100], n_outs=1, args=None):
        super(DBN, self).__init__()

        # 两个共同组成隐藏层，只是有明确分工，最后那个说明了所有隐藏层的个数
        self.sigmoid_layers = []  # 负责根据已知的参数计算
        self.rbm_layers = []
        # 负责调参
        self.n_layers = len(hidden_layer_sizes)  # 说明每个隐藏层的大小
        self.n_ins = n_ins
        self.hidden_layer_sizes = hidden_layer_sizes

        rbm1 = RBM(n_visible=self.n_ins, n_hidden=hidden_layer_sizes[0])
        rbm2 = RBM(n_visible=hidden_layer_sizes[0], n_hidden=hidden_layer_sizes[1])
        rbm3 = RBM(n_visible=hidden_layer_sizes[1], n_hidden=hidden_layer_sizes[2])
        self.rbm_layers.append(rbm1)
        self.rbm_layers.append(rbm2)
        self.rbm_layers.append(rbm3)

    def construct(self, input, i):

        if i == 0:
            self.rbm_layer_0 = self.rbm_layers[0](input=input)

        elif i == 1:
            self.rbm_layer_1 = self.rbm_layers[1](input=self.rbm_layers[0].out(input=input))

        elif i == 2:
            self.rbm_layer_2 = self.rbm_layers[2](input=self.rbm_layers[1].out(input=self.rbm_layers[0].out(input=input)))

        return self.rbm_layers[i].err


