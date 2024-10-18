from logging import getLogger
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common.initializer import XavierUniform, initializer
import model.loss as loss


class MultiLayerPerceptron(nn.Cell):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), has_bias=True, pad_mode="valid")
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), has_bias=True,pad_mode="valid")
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def construct(self, input_data: Tensor) -> Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(nn.Cell):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes      = data_feature.get('num_nodes')
        self.input_window   = config.get('input_window')
        self.output_window  = config.get('output_window')
        self.feature_dim    = data_feature.get('feature_dim', 2)
        self.output_dim     = data_feature.get('output_dim', 1)
        self.time_intervals = config.get('time_intervals')
        self._scaler        = data_feature.get('scaler')

        self.num_block           = config.get('num_block')
        self.time_series_emb_dim = config.get('time_series_emb_dim')
        self.spatial_emb_dim     = config.get('spatial_emb_dim')
        self.temp_dim_tid        = config.get('temp_dim_tid')
        self.temp_dim_diw        = config.get('temp_dim_diw')
        self.if_spatial          = config.get('if_spatial')
        self.if_time_in_day      = config.get('if_TiD')
        self.if_day_in_week      = config.get('if_DiW')


        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7
            
        if self.if_spatial:
            self.node_emb = mindspore.Parameter(initializer(XavierUniform(),
                                                          [self.num_nodes, self.spatial_emb_dim],
                                                          mindspore.float32))

        if self.if_time_in_day:
            self.time_in_day_emb = mindspore.Parameter(initializer(XavierUniform(),
                                                                  [self.time_of_day_size, self.temp_dim_tid],
                                                                  mindspore.float32))

        if self.if_day_in_week:
            self.day_in_week_emb = mindspore.Parameter(initializer(XavierUniform(),
                                                                  [self.day_of_week_size, self.temp_dim_diw],
                                                                  mindspore.float32))
    
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_window, 
            out_channels=self.time_series_emb_dim, 
            kernel_size=(1, 1),
            has_bias=True,
            pad_mode="valid")

        # encoding
        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_day_in_week) + self.temp_dim_diw * int(self.if_time_in_day)
        
        self.encoder = nn.SequentialCell(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, 
            out_channels=self.output_window, 
            kernel_size=(1, 1), 
            has_bias=True,
            pad_mode="valid")
        
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


    def forward(self, x, label):
        # prepare data
        input_data = x
        time_series:mindspore.Tensor = input_data[..., :1]

        if self.if_time_in_day:
            tid_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).long()]

        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            diw_data = ops.argmax(input_data[..., 2:], dim=-1)
            day_in_week_emb = self.day_in_week_emb[(diw_data[:, -1, :]).long()]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = time_series.shape
        time_series = time_series.swapaxes(1, 2)
        time_series = time_series.view(batch_size, num_nodes, -1).swapaxes(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series)

        node_emb = []
        if self.if_spatial:
            node_emb.append(ops.broadcast_to(self.node_emb.unsqueeze(0),(batch_size, -1, -1)).swapaxes(1, 2).unsqueeze(-1))
            

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.swapaxes(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.swapaxes(1, 2).unsqueeze(-1))

        hidden = ops.cat([time_series_emb] + node_emb + tem_emb, axis=1)  # concat all embeddings
        hidden  = self.encoder(hidden)
        prediction = self.regression_layer(hidden)

        return prediction

    def calculate_loss(self, x, label):
        y_true = label
        y_predicted, _ = self.predict(x, label)
        y_true         = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted    = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_m(y_predicted, y_true, 0)

    def predict(self, x, label):
        return self.forward(x,label), label
    
    
    def construct(self, x, label):
        x = x.astype(dtype=mindspore.float32)
        label = label.astype(dtype=mindspore.float32)
        if self.mode == "train":
            return self.calculate_loss(x, label)
        elif self.mode == "eval":        
            y_predicted,y_true = self.predict(x, label)
            y_predicted    = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
            y_true         = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            return y_predicted,y_true
