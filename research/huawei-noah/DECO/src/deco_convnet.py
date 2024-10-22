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
DECO ConvNet classes.
"""

import copy
from typing import Optional
import numpy
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as np
from mindspore import ops
from src.drop_path import DropPath2D
from src.encoder_module import *

class DECO_ConvNet(nn.Cell):
    '''DECO metaformer class'''

    def __init__(self, num_queries=100, d_model=512, enc_dims=[120,240,480], enc_depth=[2,6,2], 
                 num_decoder_layers=6, normalize_before=False, return_intermediate_dec=False, qH=10):
        super().__init__()

        # object query shape
        self.qH = qH
        self.qW = int(numpy.float(num_queries)/numpy.float(self.qH))
        print('query shape {}x{}'.format(self.qH, self.qW))  

        # encoder
        self.encoder = DecoEncoder(enc_dims=enc_dims, enc_depth=enc_depth)     

        # decoder
        decoder_layer = DecoDecoderLayer(d_model, normalize_before, self.qH, self.qW)
        decoder_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.decoder = DecoDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, qH=self.qH, qW=self.qW)

        self.tgt = nn.Embedding(num_queries, d_model)
        self.d_model = d_model

    def construct(self, src, query_embed):
        bs, c, h, w = src.shape

        query_embed = ops.Tile()(ops.ExpandDims()(query_embed, 1),(1, bs, 1))
        query_embed = ops.Transpose()(query_embed,(1, 2, 0)).view(bs, self.d_model,self.qH,self.qW)

        tgt = ops.ZerosLike()(query_embed)
        tgt = ops.Tile()(ops.ExpandDims()(self.tgt.embedding_table, 1),(1, bs, 1))

        memory = self.encoder(src)
        hs = self.decoder(tgt, memory, bs=bs, d_model=self.d_model, query_pos=query_embed)
        return hs.transpose(0,2,1,3)


class DecoEncoder(nn.Cell):
    '''Define Deco Encoder'''
    def __init__(self, enc_dims=[120,240,480], enc_depth=[2,6,2]): 
        super().__init__()
        self._encoder = ConvNeXt(depths=enc_depth, dims=enc_dims) 

    def construct(self, src):
        '''src is the last feature map of the backbone. Dimension is mapped to (h*w, bs, hidden_dim) '''   
        output = self._encoder(src)
        return output


class DecoDecoder(nn.Cell):
    '''Define Deco Decoder'''
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, qH=10, qW=10):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.qH = qH
        self.qW = qW

    def construct(self, tgt, memory, bs, d_model,
                        query_pos: Optional[Tensor] = None):
        
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = ops.Transpose()(output,(1, 2, 0)).view(bs, d_model,self.qH,self.qW)
            output = layer(output, memory, query_pos=query_pos)
            output = ops.Transpose()(output.view(bs, d_model, self.qH*self.qW),(2, 0, 1))

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            output = ops.Stack()(intermediate)
            return output
        
        return ops.ExpandDims()(output,0)


class DecoDecoderLayer(nn.Cell):
    '''Define a layer for Deco Decoder'''
    def __init__(self,d_model, normalize_before=False, qH=10, qW=10,
                 drop_path=0.,layer_scale_init_value=1e-6):
        super().__init__()
        self.normalize_before = normalize_before
        self.qH = qH
        self.qW = qW

        # The SIM module   
        self.dwconv1 = nn.Conv2d(d_model, d_model, kernel_size=9, group=d_model, has_bias=True) 
        self.norm1 = LayerNorm((d_model,), epsilon=1e-6) 
        self.pwconv1_1 = nn.Dense(d_model, 4 * d_model)
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Dense(4 * d_model, d_model)
        self.gamma1 = Parameter(Tensor(layer_scale_init_value * np.ones((d_model)), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath2D(drop_path) if drop_path > 0. else nn.Identity()

        # The CIM module
        self.dwconv2 = nn.Conv2d(d_model, d_model, kernel_size=9, group=d_model, has_bias=True) 
        self.norm2 = LayerNorm((d_model,), epsilon=1e-6)
        self.pwconv2_1 = nn.Dense(d_model, 4 * d_model) 
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Dense(4 * d_model, d_model)
        self.gamma2 = Parameter(Tensor(layer_scale_init_value * np.ones((d_model)), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath2D(drop_path) if drop_path > 0. else nn.Identity()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def construct(self, tgt, memory, 
                        query_pos: Optional[Tensor] = None):
        # SIM
        b,d,h,w = memory.shape
        tgt2 = tgt+query_pos
        tgt2 = self.dwconv1(tgt2)
        tgt2 = ops.Transpose()(tgt2,(0, 2, 3, 1)) # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = ops.Transpose()(tgt2,(0,3,1,2)) # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)

        # CIM
        tgt = ops.ResizeNearestNeighbor((h,w))(tgt) 
        tgt2 = tgt + memory 
        tgt2 = self.dwconv2(tgt2)
        tgt2 = tgt2 + tgt 
        tgt2 = ops.Transpose()(tgt2,(0, 2, 3, 1)) # (b,d,h,w)->(b,h,w,d)
        tgt2 = self.norm2(tgt2)

        # FFN
        tgt = tgt2
        tgt2 = self.pwconv2_1(tgt2)
        tgt2 = self.act2(tgt2)
        tgt2 = self.pwconv2_2(tgt2)
        if self.gamma2 is not None:
            tgt2 = self.gamma2 * tgt2
        tgt2 = ops.Transpose()(tgt2,(0,3,1,2)) # (b,h,w,d)->(b,d,h,w)
        tgt = ops.Transpose()(tgt,(0,3,1,2)) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt + self.drop_path1(tgt2)

        # pooling
        m = nn.AdaptiveMaxPool2d((self.qH,self.qW))
        tgt = m(tgt)
        
        return tgt


class LayerNorm(nn.LayerNorm):
    """LayerNorm"""
    def __init__(self, normalized_shape, epsilon, norm_axis=-1):
        super(LayerNorm, self).__init__(normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1, 1), "LayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x):
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.Transpose()(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.Transpose()(y, (0, 3, 1, 2))
        return y


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


def build_deco_convnet(args):
    return DECO_ConvNet(num_queries=args.num_queries,
                        d_model=args.hidden_dim,
                        num_decoder_layers=args.dec_layers,
                        normalize_before=args.pre_norm,
                        return_intermediate_dec=True,
                        qH=args.qH,
                        )
