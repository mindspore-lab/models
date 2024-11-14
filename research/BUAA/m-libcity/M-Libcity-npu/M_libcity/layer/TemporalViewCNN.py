import mindspore as ms
import mindspore.nn as nn

# Temporal CNN
class TemporalConvLayer(nn.Cell):
    def __init__(self, input_size, output_size, time_strides, kernel_size=3):
        """
        Args:
            input_size=d
            output_size=d'
            time_strides:
            kernel_size:
        """
        super(TemporalConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_strides = time_strides
        self.temporalCNN = nn.Conv1d(input_size, output_size, kernel_size)

    def construct(self, inputs):
        """
        Args:
            inputs:(b,t,n,d)
        Returns:
            (b,t‘,n,d')
            t'= (time_strides * t + 2 * pad - ker) / time_strides + 1
        """
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[1],inputs.shape[2]*inputs.shape[3])
        output = self.temporalCNN(inputs)
        return output