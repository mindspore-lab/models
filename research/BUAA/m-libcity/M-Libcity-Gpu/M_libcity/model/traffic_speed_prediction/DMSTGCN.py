from logging import getLogger
import mindspore as ms
import mindspore.nn as nn
import model.loss as loss
import mindspore.ops as ops
import time


class nconv(nn.Cell):
    def __init__(self):
        super(nconv, self).__init__()

    
    def construct(self, x, A):
        # TODO: result = ops.einsum('ncvl,nwv->ncwl',x, A) 报梯度回传的错
        x = x.swapaxes(2, 3)
        
        # 2. nclv -> n(c*l)v
        n, c, l, v = x.shape
        _, w, _ = A.shape
        x = x.reshape(n, c * l, v)
        
        # 3. nwv -> nvw
        A = A.swapaxes(1, 2)
        
        # 4. n(c*l)v * nvw -> n(c*l)w
        result = ops.bmm(x, A)
        
        return result.reshape(n, c, l, w).swapaxes(2,3)


class linear(nn.Cell):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=1, has_bias=True, pad_mode="valid")

    def construct(self, x):
        return self.mlp(x)


class gcn(nn.Cell):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order   = order

    def construct(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = ops.cat(out,axis=1)
        h = self.mlp(h)
        h = ops.dropout(h, self.dropout, training=self.training)
        return h


class DMSTGCN(nn.Cell):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # section 1: data_feature
        self.data_feature= data_feature
        self._scaler     = self.data_feature.get('scaler')
        self.num_nodes   = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.time_slots  = self.data_feature.get('time_slots', 288)
        self.output_dim  = self.data_feature.get('output_dim', 1)

        # section 2: model config
        self.output_window = config.get('output_window', 12)
        self._logger         = getLogger()
        self.num_layers    = config.get('num_layers', 2)
        self.dropout       = config.get('dropout', 0.3)
        self.residual_channels = config.get('residual_channels', 32)
        self.dilation_channels = config.get('dilation_channels', 32)
        self.end_channels = config.get('end_channels', 512)
        self.kernel_size  = config.get('kernel_size', 2)
        self.num_blocks   = config.get('num_blocks', 4)
        # 'days' in origin repo
        
        self.normalization  = config.get('normalization', 'batch')
        self.embedding_dims = config.get('embedding_dims', 40)
        self.order          = config.get('order', 2)
        
        self.filter_convs    = nn.CellList()
        self.gate_convs      = nn.CellList()
        self.residual_convs  = nn.CellList()
        self.skip_convs      = nn.CellList()
        self.normal          = nn.CellList()
        self.gconv           = nn.CellList()

        self.filter_convs_a  = nn.CellList()
        self.gate_convs_a    = nn.CellList()
        self.residual_convs_a= nn.CellList()
        self.skip_convs_a    = nn.CellList()
        self.normal_a        = nn.CellList()
        self.gconv_a         = nn.CellList()

        self.gconv_a2p       = nn.CellList()
        

        self.start_conv_a = nn.Conv2d(in_channels=1,
                                      out_channels=self.residual_channels,
                                      kernel_size=(1, 1),
                                      has_bias=True,
                                      pad_mode="valid")
        

        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1),
                                    has_bias=True,
                                    pad_mode="valid")
  
        self.nodevec_p1 = ms.Parameter(
            ops.randn(self.time_slots, self.embedding_dims),
            name='nodevec_p1',
            requires_grad=True
        )

        self.nodevec_p2 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_p2',
            requires_grad=True
        )


        self.nodevec_p3 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_p3',
            requires_grad=True
        )
        

        self.nodevec_pk = ms.Parameter(
            ops.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims),
            name='nodevec_pk',
            requires_grad=True
        )
        
        
        self.nodevec_a1 = ms.Parameter(
            ops.randn(self.time_slots, self.embedding_dims),
            name='nodevec_a1',
            requires_grad=True
        )

        
        self.nodevec_a2 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_a2',
            requires_grad=True
        )

        
        self.nodevec_a3 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_a3',
            requires_grad=True
        )
        

        self.nodevec_ak = ms.Parameter(
            ops.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims),
            name='nodevec_ak',
            requires_grad=True
        )

        
        self.nodevec_a2p1 = ms.Parameter(
            ops.randn(self.time_slots, self.embedding_dims),
            name='nodevec_a2p1',
            requires_grad=True
        )


        self.nodevec_a2p2 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_a2p2',
            requires_grad=True
        )

        
        self.nodevec_a2p3 = ms.Parameter(
            ops.randn(self.num_nodes, self.embedding_dims),
            name='nodevec_a2p3',
            requires_grad=True
        )

        
        self.nodevec_a2pk = ms.Parameter(
            ops.randn(self.embedding_dims, self.embedding_dims, self.embedding_dims), 
            name='nodevec_a2pk',
            requires_grad=True
        )
        
        receptive_field = 1
        skip_channels = 8
        self.supports_len = 1
        
        for _ in range(self.num_blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(self.num_layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), 
                                                   dilation=new_dilation,
                                                   has_bias=True,
                                                   pad_mode="valid"))

                
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                kernel_size=(1, self.kernel_size), 
                                                dilation=new_dilation,
                                                has_bias=True,
                                                pad_mode="valid"))

                
                
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                    kernel_size=(1, 1),
                                                    has_bias=True,
                                                    pad_mode="valid"))

                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1),
                                                has_bias=True,
                                                pad_mode="valid"))

                self.filter_convs_a.append(nn.Conv2d(in_channels=self.residual_channels,
                                                    out_channels=self.dilation_channels,
                                                    kernel_size=(1, self.kernel_size), 
                                                    dilation=new_dilation,
                                                    has_bias=True,
                                                    pad_mode="valid"))

                self.gate_convs_a.append(nn.Conv2d(in_channels=self.residual_channels,
                                                    out_channels=self.dilation_channels,
                                                    kernel_size=(1, self.kernel_size), 
                                                    dilation=new_dilation,
                                                    has_bias=True,
                                                    pad_mode="valid"))
                
                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                        out_channels=self.residual_channels,
                                                        kernel_size=(1, 1),
                                                        has_bias=True,
                                                        pad_mode="valid"))
                
                if self.normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(self.residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(self.residual_channels))
                elif self.normalization == "layer":
                    self.normal.append(
                        nn.LayerNorm([self.residual_channels, self.num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(
                        nn.LayerNorm([self.residual_channels, self.num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation     *= 2
                receptive_field  += additional_scope
                additional_scope *= 2
                
                
                self.gconv.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))
                
                self.gconv_a.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))
                
                self.gconv_a2p.append(
                    gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len,
                        order=self.order))
                

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    has_bias=True,
                                    pad_mode="valid")

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    has_bias=True,
                                    pad_mode="valid")

        self.receptive_field = receptive_field
        for i, param in enumerate(self.trainable_params()):
            param.name = f"{param.name}_{i}"

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = ops.einsum("ai,ijk->ajk", time_embedding, core_embedding)
        adp = ops.einsum("bj,ajk->abk", source_embedding, adp)
        adp = ops.einsum("ck,abk->abc", target_embedding, adp)
        adp = ops.softmax(ops.relu(adp), axis=2)
        return adp
    
    def train(self):
        self.mode = "train"
        self.set_grad(True)
        self.set_train(True)

    def eval(self):
        self.mode = "eval"
        self.set_grad(False)
        self.set_train(False)
        
    def validate(self):
        self.set_grad(False)
        self.set_train(False)
        
    def construct(self, x, label,idx:ms.Tensor):
        x     = x.astype(dtype=ms.float32)
        label = label.astype(dtype=ms.float32)
        idx = idx.int()
        if self.mode == "train":
            loss = self.calculate_loss(x, label, idx)
            return loss
        elif self.mode == "eval":
            y_predicted,y_true = self.predict(x, label, idx)
            y_predicted        = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
            y_true             = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            return y_predicted,y_true
        

    def predict(self, x, y_true, idx):
        """
        input:(B,T,N,F)-> (B, F, N, T)
        其中F包含两个特征，第0个是主特征，第1个是辅助特征。因此本模型只适合PEMSD4和8数据集
        论文中分别使用speed/flow作为主/辅助特征。使用其他特征，需要修改raw_data/dataset_name/config.json文件中的"data_col"属性。
        """
        inputs = x
        inputs = inputs.permute(0, 3, 2, 1)
        in_len = inputs.shape[3]
        if in_len < self.receptive_field:
            xo = ops.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        # xo[:,[0]] means primary feature
        x = self.start_conv(xo[:, [0]])
        # xo[:,[1]] means auxiliary feature, witch can be set in the raw_data/dataset_name/config.json file
        x_a = self.start_conv_a(xo[:, [1]])
        skip = 0
   
        adp = self.dgconstruct(self.nodevec_p1[idx], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adp_a = self.dgconstruct(self.nodevec_a1[idx], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        adp_a2p = self.dgconstruct(self.nodevec_a2p1[idx], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

        new_supports     = [adp]
        new_supports_a   = [adp_a]
        new_supports_a2p = [adp_a2p]
        

        for i in range(self.num_blocks * self.num_layers):
            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = ops.tanh(filter)
            gate   = self.gate_convs[i](residual)
            gate   = ops.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            filter_a  = self.filter_convs_a[i](residual_a)
            filter_a  = ops.tanh(filter_a)
            gate_a    = self.gate_convs_a[i](residual_a)
            gate_a    = ops.sigmoid(gate_a)
            x_a       = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.swapaxes(2, 3).reshape([s.shape[0], -1, s.shape[2], 1])
            else:
                skip = ops.cat([s.swapaxes(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], axis=1)

            # dynamic graph convolutions
            x   = self.gconv[i](x, new_supports)
            
            x_a = self.gconv_a[i](x_a, new_supports_a)

            # multi-faceted fusion module
            x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            x     = x_a2p + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.shape[3]:]
            x   = x + residual[:, :, :, -x.shape[3]:]
            x   = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = ops.relu(skip)
        x = ops.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        y_true = y_true[..., 2:3]  
        return x,y_true


    def calculate_loss(self, x, y, idx):
        y_predicted,y_true = self.predict(x,y,idx)
        y_true      = self._scaler.inverse_transform(y_true[..., : self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_m(y_predicted, y_true)
