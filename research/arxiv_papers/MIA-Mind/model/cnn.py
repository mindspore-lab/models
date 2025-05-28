import mindspore.nn as nn
import mindspore.ops as ops
from MIA_Mind import MIA

# -------------------- CNN for MachineLearningCVE with MIAM --------------------

class CNN_mia(nn.Cell):
    def __init__(self, in_channels):
        super(CNN_mia, self).__init__()
        self.reshape = ops.Reshape()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 3), stride=1, pad_mode="valid")
        self.relu1 = nn.ReLU()
        self.cbam1 = MIA(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, pad_mode="valid")
        self.relu2 = nn.ReLU()
        self.cbam2 = MIA(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(4608, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Dense(128, 2)

    def construct(self, x):
        x = self.reshape(x, (x.shape[0], 1, 1, -1))
        x = self.pool1(self.cbam1(self.relu1(self.conv1(x))))
        x = self.pool2(self.cbam2(self.relu2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)
