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
"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

"""

from src.Utils import *

#指定下层向量2维，上层向量3维，一旦有输入矩阵，就会多次上下调参，直到找到计算出新形式的映射函数。
#这就与hiddenlayer不同，那个只负责计算，不负责调参
class RBM(nn.Cell):

    def __init__(self, input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, numpy_rng=None):
        super(RBM, self).__init__()
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer

        if W is None:
            initial_W = ops.Zeros()((n_hidden, n_visible), mindspore.float32)
            W = initial_W

        if hbias is None:
            hbias = ops.Zeros()((n_hidden,), mindspore.float32)  # initialize h bias 0

        if vbias is None:
            vbias = ops.Zeros()((n_visible,), mindspore.float32)  # initialize v bias 0

        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.W_ = ops.Zeros()((n_hidden, n_visible), mindspore.float32)
        self.hbias_ = ops.Zeros()((n_hidden,), mindspore.float32)
        self.vbias_ = ops.Zeros()((n_visible,), mindspore.float32)

        # self.params = [self.W, self.hbias, self.vbias]

    # 微调，实际上也可以看成一种调参。与有监督不同，有监督使用数据和标签调参，无监督用数据和数据本身的新形式调参。本质是一样的
    def construct(self, lr=0.1, momentum=0.5, k=1, input=None):
        if input is not None:
            self.input = input

        ''' CD-k '''
        ph_mean = self.sample_h_given_v(self.input)  # h1

        chain_start = ph_mean

        # 上下来回k次互相生成矩阵
        for step in range(k):  # nv_means:基于v计算h的条件概率 nv_samples:二项分布采样结果
            # nh_means:基于h计算v的条件概率 nh_samples:二项分布采样结果
            if step == 0:
                nv_means, \
                nh_means, = self.gibbs_hvh(chain_start)
            else:
                nv_means, \
                nh_means = self.gibbs_hvh(nh_means)

        # 调w和b
        self.W_ = (momentum * self.W_) + lr * (ops.dot(ph_mean.T, self.input) - ops.dot(nh_means.T, nv_means)) / \
                  self.input.shape[0]  # self.input:v1  nv_means:v2
        # ph_mean:h1  nh_means:h2
        self.vbias_ = (momentum * self.vbias_) + lr * (ops.ReduceMean()(self.input - nv_means, axis=0))

        self.hbias_ = (momentum * self.hbias_) + lr * (ops.ReduceMean()(ph_mean - nh_means, axis=0))

        self.W = self.W_ + self.W
        self.vbias = self.vbias_ + self.vbias
        self.hbias = self.hbias_ + self.hbias

        self.err = ops.ReduceSum()((self.input - nv_means) ** 2) / self.input.shape[0]

        return self.out(self.input)

    # 将下面的矩阵整合计算，然后得到上面矩阵
    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)

        return h1_mean

    # 将上面的矩阵整合计算，然后得到下面矩阵
    def sample_v_given_h(self, h0_sample):
        # h算v
        v1_mean = self.propdown(h0_sample)

        return v1_mean

    def propup(self, v):
        pre_sigmoid_activation = ops.dot(v, self.W.T) + self.hbias
        return sigmrnd_tensor(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = ops.dot(h, self.W) + self.vbias
        return sigmrnd_tensor(pre_sigmoid_activation)

    # 先上调下，再下调上
    def gibbs_hvh(self, h0_sample):
        v1_mean = self.sample_v_given_h(h0_sample)  # v2
        h1_mean = self.sample_h_given_v(v1_mean)  # h2

        return [v1_mean,
                h1_mean, ]

    def out(self, input):
        pre_sigmoid_activation = (ops.dot(input, self.W.T) + self.hbias)
        out = sigmoid_tensor(pre_sigmoid_activation)
        return out
