# -*- encoding: utf-8 -*-
# here put the import lib
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from diffusion.utils import get_timestep_embedding, nonlinearity, Normalize



class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = mindspore.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        has_bias=True, 
                                        pad_mode='pad')

    def construct(self, x):
        x = ops.interpolate(
            x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = mindspore.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0,
                                        has_bias=True, 
                                        pad_mode='pad')

    def construct(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = mindspore.ops.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = mindspore.ops.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Cell):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = mindspore.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     has_bias=True, 
                                     pad_mode='pad')
        self.temb_proj = mindspore.nn.Dense(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = mindspore.nn.Dropout(dropout)
        self.conv2 = mindspore.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     has_bias=True, 
                                     pad_mode='pad')
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = mindspore.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     has_bias=True, 
                                                     pad_mode='pad')
            else:
                self.nin_shortcut = mindspore.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    has_bias=True, 
                                                    pad_mode='pad')
    
    def construct(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = mindspore.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 has_bias=True, 
                                 pad_mode='pad')
        self.k = mindspore.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 has_bias=True, 
                                 pad_mode='pad')
        self.v = mindspore.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 has_bias=True, 
                                 pad_mode='pad')
        self.proj_out = mindspore.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        has_bias=True, 
                                        pad_mode='pad')

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = ops.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = ops.softmax(w_, axis=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class UNet(nn.Cell):
    def __init__(self, config, guide=False):
        super().__init__()
        self.config = config
        self.guide = guide
        # ch is the embedding size, out_ch is out size, ch_mult is the list of multiple of last layer size
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        #resolution = config.data.image_size
        resolution = int(math.sqrt(config.hidden_size))
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        # if config.model.type == 'bayesian':
        #     self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult) # 上采样以及下采样几次
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.hidden_size = config.hidden_size

        # timestep embedding
        self.temb = nn.Cell()
        self.temb.dense = nn.CellList([
            mindspore.nn.Dense(self.ch,
                            self.temb_ch),
            mindspore.nn.Dense(self.temb_ch,
                            self.temb_ch),
        ])

        # guide vector
        if self.guide:
            self.guide_emb = nn.Cell()
            self.guide_emb.dense = nn.CellList([
                mindspore.nn.Dense(self.hidden_size,
                                self.temb_ch),
                mindspore.nn.Dense(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = mindspore.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       has_bias=True, 
                                       pad_mode='pad')

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult   # (1, ch_multi)
        self.down = nn.CellList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = ch*in_ch_mult[i_level]   # num of input channel
            block_out = ch*ch_mult[i_level] # num of output channel
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = mindspore.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        has_bias=True, 
                                        pad_mode='pad')

    def construct(self, x, t, guide):
        #assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb) # (bs, 4 * hidden_size)

        # get the guide vector
        if self.guide:
            guide_emb = self.guide_emb.dense[0](guide)  # (bs, hidden_size)
            guide_emb = nonlinearity(guide_emb)
            guide_emb = self.guide_emb.dense[1](guide_emb) # (bs, 4 * hidden_size)

            temb += guide_emb

        original_shape = x.shape
        x = ops.reshape(x, [x.shape[0], -1, self.resolution, self.resolution])

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))    # 缩减维度

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    ops.cat([h, hs.pop()], axis=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        h = ops.reshape(h, original_shape)

        return h





