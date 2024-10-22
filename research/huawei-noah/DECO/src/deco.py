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

"""
DECO model classes.
"""

import math
import copy
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as init
from src.init_weights import KaimingUniform
from src.init_weights import UniformBias
from src.backbone_resnet import Backbone,resnet50
from src.deco_convnet import build_deco_convnet

def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])

class DECO(nn.Cell):
    """ This is the DECO module that performs object detection """
    def __init__(self, backbone, deco_convnet, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: the module of backbone. See backbone_resnet.py
            deco_convnet: the module of the deco_convnet architecture, including encoder and decoder. See deco_convnet.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DECO can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.deco_convnet = deco_convnet
        hidden_dim = deco_convnet.d_model
        self.class_embed = nn.Dense(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) 
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.aux_loss = aux_loss

        # backbone
        self.backbone = backbone 
        
        # input projection 
        print('backbone.num_channels = {}'.format(backbone.num_channels))
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], 120, kernel_size=1, has_bias=True)

        # not shared head
        two_stage=False
        with_box_refine=True
        num_pred = (deco_convnet.decoder.num_layers + 1) if two_stage else deco_convnet.decoder.num_layers        
        
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        init.Constant(value=bias_value)(self.class_embed.bias.data) 

        # init bbox_mebed
        init.Constant(value=0)(self.bbox_embed.layers[-1].weight.data)
        init.Constant(value=0)(self.bbox_embed.layers[-1].bias.data)
        init.Constant(value=-2.0)(self.bbox_embed.layers[-1].bias.data[2:])
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            init.Constant(value=-2.0)(self.bbox_embed[0].layers[-1].bias.data[2:])
            # hack implementation for iterative bounding box refinement
            self.deco_convnet.decoder.bbox_embed = self.bbox_embed
        else:
            init.Constant(value=-2.0)(self.bbox_embed.layers[-1].bias.data[2:])
            self.class_embed = nn.CellList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.CellList([self.bbox_embed for _ in range(num_pred)])
            self.deco_convnet.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.deco_convnet.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                init.Constant(value=0.0)(box_embed.layers[-1].bias.data[2:])

    def construct(self, inputs):
        """construct"""

        features = self.backbone(inputs)
        src = features[-1]["data"]

        hs = self.deco_convnet(self.input_proj(src), self.query_embed.embedding_table) 
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            outputs_coord = ops.Sigmoid()(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = ops.stack(outputs_classes)
        outputs_coord = ops.stack(outputs_coords)

        if self.aux_loss:
            output = ops.Concat(axis=-1)([outputs_class, outputs_coord])
        else:
            output = ops.Concat(axis=-1)([outputs_class[-1], outputs_coord[-1]])
        return output


class MLP(nn.Cell):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([
            nn.Dense(n, k, weight_init=KaimingUniform(), bias_init=UniformBias([k, n]))
            for n, k in zip([input_dim] + h, h + [output_dim])
        ])
        
    def construct(self, x):
        """construct"""
        for i, layer in enumerate(self.layers):
            x = ops.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_deco(args):
    '''build DECO model'''
    num_classes = args.num_classes

    backbone = Backbone(resnet50())
    deco_convnet = build_deco_convnet(args)

    model = DECO(
        backbone,
        deco_convnet,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )

    return model

