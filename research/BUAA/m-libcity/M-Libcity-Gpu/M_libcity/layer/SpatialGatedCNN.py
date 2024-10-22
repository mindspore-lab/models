  
import mindspore.nn as nn
import mindspore.ops as ops

class SpatialGatedCNN(nn.Cell):
    def __init__(self,input_size,output_size,kernal_size=(3,3),padding=1):
        super(SpatialGatedCNN, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.cnn1=nn.Conv2d(input_size,output_size,kernal_size,padding=padding, pad_mode='pad')
        self.cnn2 = nn.Conv2d(input_size, output_size, kernal_size, padding=padding, pad_mode='pad')
        self.sigmoid = ops.Sigmoid()
        self.relu = ops.ReLU()
    
    def construct(self, inputs):
        """
        Args:
            inputs: shape = (b,c_in,t,N)
        Returns:
            (b,c_out,t,N)
        """
        gate=self.sigmoid(self.relu(self.cnn1(inputs)))
        value=self.relu(self.cnn2(inputs))

        return gate*value