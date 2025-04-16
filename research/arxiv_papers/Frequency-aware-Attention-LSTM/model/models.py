import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import XavierUniform

class AdaptiveFrequencyDecomposition(nn.Cell):
    """频率分解模块"""
    def __init__(self):
        super().__init__()
        self.convs = nn.CellList([
            nn.SequentialCell(
                nn.ConstantPad1d((size - 1, 0), 0),
                nn.Conv1d(1, 1, kernel_size=size, weight_init=XavierUniform(), pad_mode='valid'),
                nn.Tanh()
            ) for size in [16, 8, 4, 2]
        ])
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        x = ops.transpose(x, (0, 2, 1))
        features = [conv(x) for conv in self.convs]
        combined = ops.cat(features, axis=1)
        weights = self.softmax(combined.mean(axis=2, keep_dims=True))
        return ops.transpose(combined * weights, (0, 2, 1))

class PM25PredictionModel(nn.Cell):
    """PM2.5预测主模型"""
    def __init__(self, feat_dim, hidden_dim=32, num_heads=2):
        super().__init__()
        self.decomposers = nn.CellList([AdaptiveFrequencyDecomposition() for _ in range(feat_dim)])
        self.lstm = nn.LSTM(feat_dim * 4, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads)
        self.dropout2 = nn.Dropout(p=0.3)
        self.norm = nn.LayerNorm((hidden_dim * 2,))
        self.fc = nn.SequentialCell(
            nn.Dense(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Dense(32, 1)
        )

    def construct(self, x):
        decomposed = []
        for i in range(x.shape[2]):
            decomposed.append(self.decomposers[i](x[:, :, i:i + 1]))

        lstm_in = ops.cat(decomposed, axis=2)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout1(lstm_out)

        lstm_out = ops.transpose(lstm_out, (1, 0, 2))
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout2(attn_out)
        res_out = self.norm(lstm_out + attn_out)
        res_out = ops.transpose(res_out, (1, 0, 2))

        return self.fc(res_out[:, -1, :])