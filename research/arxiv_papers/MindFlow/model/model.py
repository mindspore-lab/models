import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class CyberDefenseModel(nn.Cell):
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.SequentialCell(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=5)
        )

        self.lstm = nn.LSTM(32, 64, bidirectional=True, batch_first=True)
        self.threshold = ms.Parameter(Tensor([0.6], ms.float32), requires_grad=False)
        self.fc = nn.SequentialCell(
            nn.Dense(128, 32),
            nn.ReLU(),
            nn.Dense(32, 1)
        )

    def construct(self, x):
        x = ops.transpose(x, (0, 2, 1))
        cnn_out = self.cnn(x)
        cnn_out = ops.transpose(cnn_out, (0, 2, 1))
        lstm_out, _ = self.lstm(cnn_out)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out).squeeze()
        return ops.sigmoid(logits), self.threshold