# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import XavierNormal
from typing import Dict, Tuple, Optional, List

MAKE_CONVERT_COMPATIBLE = os.environ.get("TFT_SCRIPTING", None) is not None

class MaybeLayerNorm(nn.Cell):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = nn.LayerNorm([output_size if output_size else hidden_size], epsilon=eps)
    
    def construct(self, x):
        return self.ln(x)


class GLU(nn.Cell):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Dense(hidden_size, output_size * 2)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.lin(x)
        x = ms.ops.glu(x)
        return x

class GRN(nn.Cell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0.0,):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Dense(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Dense(context_hidden_size, hidden_size)
        else:
            self.lin_c = nn.Identity()
        self.lin_i = nn.Dense(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Dense(input_size, output_size) if output_size else None

    def construct(self, a: ms.Tensor, c: Optional[ms.Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = ms.ops.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        if self.out_proj is None:
            y = a
        else:
            y = self.out_proj(a)
        # y = a if self.out_proj is None else self.out_proj(a)
        x = x + y
        return self.layer_norm(x) 


def fused_pointwise_linear_v1(x, a, b):
    out = ms.ops.mul(x.unsqueeze(-1), a)
    out = out + b
    return out

def fused_pointwise_linear_v2(x, a, b):
    out = x.unsqueeze(3) * a
    out = out + b
    return out


class TFTEmbedding(nn.Cell):
    def __init__(self, config, initialize_cont_params=True):
        # initialize_cont_params=False prevents form initializing parameters inside this class
        # so they can be lazily initialized in LazyEmbedding module
        super().__init__()
        self.s_cat_inp_lens    = config.static_categorical_inp_lens
        self.t_cat_k_inp_lens  = config.temporal_known_categorical_inp_lens
        self.t_cat_o_inp_lens  = config.temporal_observed_categorical_inp_lens
        self.s_cont_inp_size   = config.static_continuous_inp_size
        self.t_cont_k_inp_size = config.temporal_known_continuous_inp_size
        self.t_cont_o_inp_size = config.temporal_observed_continuous_inp_size
        self.t_tgt_size        = config.temporal_target_size

        self.hidden_size = config.hidden_size

        # There are 7 types of input:
        # 1. Static categorical
        # 2. Static continuous
        # 3. Temporal known a priori categorical
        # 4. Temporal known a priori continuous
        # 5. Temporal observed categorical
        # 6. Temporal observed continuous
        # 7. Temporal observed targets (time series obseved so far)

        self.s_cat_embed = nn.CellList([
            nn.Embedding(n, self.hidden_size) for n in self.s_cat_inp_lens]) if self.s_cat_inp_lens else None
        self.t_cat_k_embed = nn.CellList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_k_inp_lens]) if self.t_cat_k_inp_lens else None
        self.t_cat_o_embed = nn.CellList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_o_inp_lens]) if self.t_cat_o_inp_lens else None

        if initialize_cont_params:
            self.s_cont_embedding_vectors = ms.Parameter(ms.Tensor(shape=(self.s_cont_inp_size, self.hidden_size), dtype=ms.float32, init=XavierNormal()))if self.s_cont_inp_size else None
            self.t_cont_k_embedding_vectors = ms.Parameter(ms.Tensor(shape=(self.t_cont_k_inp_size, self.hidden_size), dtype=ms.float32, init=XavierNormal())) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_vectors = ms.Parameter(ms.Tensor(shape=(self.t_cont_o_inp_size, self.hidden_size), dtype=ms.float32, init=XavierNormal())) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_vectors = ms.Parameter(ms.Tensor(shape=(self.t_tgt_size, self.hidden_size), dtype=ms.float32, init=XavierNormal()))

            self.s_cont_embedding_bias = ms.Parameter(ms.ops.zeros((self.s_cont_inp_size, self.hidden_size), ms.float32)) if self.s_cont_inp_size else None
            self.t_cont_k_embedding_bias = ms.Parameter(ms.ops.zeros((self.t_cont_k_inp_size, self.hidden_size), ms.float32)) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_bias = ms.Parameter(ms.ops.zeros((self.t_cont_o_inp_size, self.hidden_size), ms.float32)) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_bias = ms.Parameter(ms.ops.zeros((self.t_tgt_size, self.hidden_size), ms.float32))

    def _apply_embedding(self,
            cat: Optional[ms.Tensor],
            cont: Optional[ms.Tensor],
            cat_emb: Optional[nn.CellList], 
            cont_emb: ms.Tensor,
            cont_bias: ms.Tensor,
            ) -> Tuple[Optional[ms.Tensor], Optional[ms.Tensor]]:
        cast = ms.ops.Cast()
        e_cat = ms.ops.stack([embed(cast(cat[...,i], ms.int32)) for i, embed in enumerate(cat_emb)], axis=-2) if cat is not None else None
        if cont is not None:
            if MAKE_CONVERT_COMPATIBLE:
                e_cont = ms.ops.mul(cont.unsqueeze(-1), cont_emb)
                e_cont = e_cont + cont_bias
            else:
                e_cont = fused_pointwise_linear_v1(cont, cont_emb, cont_bias)
        else:
            e_cont = None

        if e_cat is not None and e_cont is not None:
            return ms.ops.cat([e_cat, e_cont], axis=-2)
        elif e_cat is not None:
            return e_cat
        elif e_cont is not None:
            return e_cont
        else:
            return None

    def construct(self, x: Dict[str, ms.Tensor]):
        # temporal/static categorical/continuous known/observed input 
        s_cat_inp = x.get('s_cat', None)
        s_cont_inp = x.get('s_cont', None)
        t_cat_k_inp = x.get('k_cat', None)
        t_cont_k_inp = x.get('k_cont', None)
        t_cat_o_inp = x.get('o_cat', None)
        t_cont_o_inp = x.get('o_cont', None)
        t_tgt_obs = x['target'] # Has to be present

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        s_cat_inp = s_cat_inp[:,0,:] if s_cat_inp is not None else None
        s_cont_inp = s_cont_inp[:,0,:] if s_cont_inp is not None else None

        s_inp = self._apply_embedding(s_cat_inp,
                                      s_cont_inp,
                                      self.s_cat_embed,
                                      self.s_cont_embedding_vectors,
                                      self.s_cont_embedding_bias)
        t_known_inp = self._apply_embedding(t_cat_k_inp,
                                            t_cont_k_inp,
                                            self.t_cat_k_embed,
                                            self.t_cont_k_embedding_vectors,
                                            self.t_cont_k_embedding_bias)
        t_observed_inp = self._apply_embedding(t_cat_o_inp,
                                               t_cont_o_inp,
                                               self.t_cat_o_embed,
                                               self.t_cont_o_embedding_vectors,
                                               self.t_cont_o_embedding_bias)

        # Temporal observed targets
        if MAKE_CONVERT_COMPATIBLE:
            t_observed_tgt = ms.ops.matmul(t_tgt_obs.unsqueeze(3).unsqueeze(4), self.t_tgt_embedding_vectors.unsqueeze(1)).squeeze(3)
            t_observed_tgt = t_observed_tgt + self.t_tgt_embedding_bias
        else:
            t_observed_tgt = fused_pointwise_linear_v2(t_tgt_obs, self.t_tgt_embedding_vectors, self.t_tgt_embedding_bias)

        return s_inp, t_known_inp, t_observed_inp, t_observed_tgt

class VariableSelectionNetwork(nn.Cell):
    def __init__(self, config, num_inputs):
        super().__init__()
        self.joint_grn = GRN(config.hidden_size*num_inputs, config.hidden_size, output_size=num_inputs, context_hidden_size=config.hidden_size)
        self.var_grns = nn.CellList([GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(num_inputs)])

    def construct(self, x: ms.Tensor, context: Optional[ms.Tensor] = None):
        Xi = ms.ops.flatten(x, start_dim=-2)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = ms.ops.softmax(grn_outputs, axis=-1)
        transformed_embed_list = [m(x[...,i,:]) for i, m in enumerate(self.var_grns)]
        transformed_embed = ms.ops.stack(transformed_embed_list, axis=-1)
        #the line below performs batched matrix vector multiplication
        #for temporal features it's bthf,btf->bth
        #for static features it's bhf,bf->bh
        variable_ctx = ms.ops.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)

        return variable_ctx, sparse_weights

class StaticCovariateEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.vsn = VariableSelectionNetwork(config, config.num_static_vars)
        self.context_grns = nn.CellList([GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(4)])

    def construct(self, x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        cs, ce, ch, cc = [m(variable_ctx) for m in self.context_grns]

        return cs, ce, ch, cc


class InterpretableMultiHeadAttention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        assert config.hidden_size % config.n_head == 0
        self.d_head = config.hidden_size // config.n_head
        self.qkv_linears = nn.Dense(config.hidden_size, (2 * self.n_head + 1) * self.d_head)
        self.out_proj = nn.Dense(self.d_head, config.hidden_size)
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.out_dropout = nn.Dropout(p=config.dropout)
        self.scale = self.d_head**-0.5
        self._mask = ms.ops.triu(ms.ops.full((config.example_length, config.example_length), float('-inf')), diagonal=1).unsqueeze(0)

    def construct(self, x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), axis=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        attn_score = ms.ops.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score = ms.ops.mul(attn_score, self.scale)

        attn_score = attn_score + self._mask

        attn_prob = ms.ops.softmax(attn_score, axis=3)
        attn_prob = self.attn_dropout(attn_prob)
        # attn_vec = ms.ops.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = ms.ops.matmul(attn_prob, ms.ops.tile(v.unsqueeze(1), (1, attn_prob.shape[1], 1, 1)))
        m_attn_vec = ms.ops.mean(attn_vec, axis=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_prob

class TFTBack(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.encoder_length = config.encoder_length
        self.history_vsn = VariableSelectionNetwork(config, config.num_historic_vars)
        self.history_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(config, config.num_future_vars)
        self.future_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)


        self.input_gate = GLU(config.hidden_size, config.hidden_size)
        self.input_gate_ln = nn.LayerNorm([config.hidden_size], epsilon=1e-3)

        self.enrichment_grn = GRN(config.hidden_size,
                                  config.hidden_size,
                                  context_hidden_size=config.hidden_size, 
                                  dropout=config.dropout)
        self.attention = InterpretableMultiHeadAttention(config)
        self.attention_gate = GLU(config.hidden_size, config.hidden_size)
        self.attention_ln = nn.LayerNorm([config.hidden_size], epsilon=1e-3)

        self.positionwise_grn = GRN(config.hidden_size,
                                    config.hidden_size,
                                    dropout=config.dropout)

        self.decoder_gate = GLU(config.hidden_size, config.hidden_size)
        self.decoder_ln = nn.LayerNorm([config.hidden_size], epsilon=1e-3)

        self.quantile_proj = nn.Dense(config.hidden_size, len(config.quantiles))
        
    def construct(self, historical_inputs, cs, ch, cc, ce, future_inputs):
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc)) #令人迷惑的Ascend LSTM
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)
        # skip connection
        input_embedding = ms.ops.cat([historical_features, future_features], axis=1)
        temporal_features = ms.ops.cat([history, future], axis=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)
        # Static enrichment
        enriched = self.enrichment_grn(temporal_features, c=ce)

        # Temporal self attention
        x, _ = self.attention(enriched)

        # Don't compute hictorical quantiles
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]
        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)

        # Position-wise feed-construct
        x = self.positionwise_grn(x)
        # Final skip connection
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)
        out = self.quantile_proj(x)

        return out


class TemporalFusionTransformer(nn.Cell):
    """ 
    Implementation of https://arxiv.org/abs/1912.09363 
    """
    def __init__(self, config):
        super().__init__()

        if hasattr(config, 'model'):
            config = config.model

        self.encoder_length = config.encoder_length #this determines from how distant past we want to use data from

        self.embedding = TFTEmbedding(config)
        self.static_encoder = StaticCovariateEncoder(config)
        self.TFTpart2 = TFTBack(config)

    def construct(self, x: Dict[str, ms.Tensor]) -> ms.Tensor:
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(x)

        # Static context
        cs, ce, ch, cc = self.static_encoder(s_inp)
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0) #lstm initial states
        # Temporal input
        _historical_inputs = [t_known_inp[:,:self.encoder_length,:], t_observed_tgt[:,:self.encoder_length,:]]
        if t_observed_inp is not None:
            _historical_inputs.insert(0,t_observed_inp[:,:self.encoder_length,:])
        historical_inputs = ms.ops.concat(_historical_inputs, axis=-2)
        future_inputs = t_known_inp[:, self.encoder_length:]
        return self.TFTpart2(historical_inputs, cs, ch, cc, ce, future_inputs)

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..data.data_utils import InputTypes, DataTypes, FeatureSpec
import datetime

class ElectricityConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('power_usage', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('hour', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'days_from_start' # This column contains time indices across which we split the data
        self.train_range = (1096, 1315)
        self.valid_range = (1308, 1339)
        self.test_range = (1332, 1346)
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [369]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.1
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


class TrafficConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('values', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('time_on_day', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'sensor_day' # This column contains time indices across which we split the data
        self.train_range = (0, 151)
        self.valid_range = (144, 166)
        self.test_range = (159, float('inf'))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = False
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [963]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.3
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


CONFIGS = {'electricity':  ElectricityConfig,
           'traffic':      TrafficConfig,
           }
