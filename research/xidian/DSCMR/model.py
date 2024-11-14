from mindspore import nn , ops
from mindspore.common.initializer import Normal
from pdb import set_trace
from mindspore import dtype as mstype




class ImgNN(nn.Cell):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Dense(input_dim, output_dim)
        self.relu=ops.ReLU()
    def construct(self, x):
        #set_trace()
        x1 = self.denseL1(x)
        out = self.relu(x1)
        return out

class TextNN(nn.Cell):
    """Network to learn image representations"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Dense(input_dim, output_dim)
        self.relu = ops.ReLU()
    def construct(self, x):
        x1=self.denseL1(x)
        out=self.relu(x1)
        return out

class IDCM_NN(nn.Cell):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.text_net = TextNN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Dense(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Dense(minus_one_dim, output_dim)
    def construct(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)
        #set_trace()
        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict





