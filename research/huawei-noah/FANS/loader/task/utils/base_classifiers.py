# Copyright 2023 Huawei Technologies Co., Ltd
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


import mindspore
from mindspore import nn


class TransformLayer(nn.Cell):
    def __init__(
            self,
            hidden_size,
            activation_function=None,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Dense(hidden_size, hidden_size)
        if layer_norm_eps is None:
            self.LayerNorm = nn.LayerNorm(hidden_size)
        else:
            self.LayerNorm = nn.LayerNorm((hidden_size,), epsilon=layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = mindspore.ops.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Cell):
    def __init__(
            self,
            hidden_size,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        # self.decoder = nn.Dense(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bias = mindspore.Parameter(mindspore.ops.zeros(vocab_size), requires_grad=True)
        # self.decoder.bias = self.bias

    def construct(self, hidden_states):
        self.decoder = nn.Dense(self.hidden_size, self.vocab_size, has_bias=False)
        self.decoder.bias = self.bias
        return self.decoder(hidden_states)


class BaseClassifier(nn.Cell):
    classifiers = dict()

    def __init__(
            self,
            vocab_size,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(BaseClassifier, self).__init__()

        self.transform_layer = TransformLayer(
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder_layer = DecoderLayer(
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )

    @classmethod
    def create(cls, key, vocab_size, hidden_size, activation_function, layer_norm_eps=None):
        if key in cls.classifiers:
            return cls.classifiers[key]

        classifier = cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        cls.classifiers[key] = classifier
        return classifier

    def construct(self, last_hidden_states):
        hidden_states = self.transform_layer(last_hidden_states)
        prediction = self.decoder_layer(hidden_states)
        return prediction


class BertClassifier(BaseClassifier):
    @classmethod
    def create(
            cls,
            config,
            key,
            vocab_size,
            **kwargs
    ):
        return super().create(
            key=key,
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            activation_function=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )


class ClusterClassifier(nn.Cell):
    classifiers = dict()

    def __init__(
            self,
            cluster_vocabs,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(ClusterClassifier, self).__init__()
        self.n_clusters = len(cluster_vocabs)
        self.hidden_size = hidden_size

        self.transform_layer = TransformLayer(
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )

        self.decoder_layers = nn.CellList([
            DecoderLayer(
                hidden_size=hidden_size,
                vocab_size=vocab_size
            ) for vocab_size in cluster_vocabs
        ])

    @classmethod
    def create(cls, key, cluster_vocabs, hidden_size, activation_function, layer_norm_eps=None):
        if key in cls.classifiers:
            return cls.classifiers[key]

        classifier = cls(
            cluster_vocabs=cluster_vocabs,
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        cls.classifiers[key] = classifier
        return classifier

    def construct(self, last_hidden_states, cluster_labels):
        """

        :param cluster_labels: torch.Tensor([batch_size, sequence_length])
        :param last_hidden_states: torch.Tensor([batch_size, sequence_length, hidden_size])
        """

        hidden_states = self.transform_layer(last_hidden_states)  # type:
        predictions = []

        for i_cluster in range(self.n_clusters):
            # print('i_cluster',i_cluster)
            # print('cluster_labels',cluster_labels)
            mask = mindspore.ops.equal(cluster_labels, i_cluster).unsqueeze(dim=-1)
            # mask = (cluster_labels == i_cluster).unsqueeze(dim=-1)
            if not mask.sum():
                predictions.append(None)
            else:
                cluster_hidden_states = mindspore.ops.masked_select(hidden_states, mask).reshape(-1, self.hidden_size)  # [L, D]
                predictions.append(self.decoder_layers[i_cluster](cluster_hidden_states))  # [L, V]

        return predictions


class BertClusterClassifier(ClusterClassifier):
    @classmethod
    def create(cls, key, cluster_vocabs, config, **kwargs):
        return super().create(
            key=key,
            cluster_vocabs=cluster_vocabs,
            hidden_size=config.hidden_size,
            activation_function=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )
