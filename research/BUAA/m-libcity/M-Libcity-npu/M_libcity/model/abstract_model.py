import mindspore.nn as nn

class AbstractModel(nn.Cell):

    def __init__(self, config, data_feature):
        nn.Cell.__init__(self)

    def predict(self,batch):
        """
        根据具体模型需求实现预测，主要用于test阶段


        Args:
            batch (Tensor): inputs
        Returns:
            tensor: predict result of this batch

        """
        return NotImplementedError("predict not implemented")