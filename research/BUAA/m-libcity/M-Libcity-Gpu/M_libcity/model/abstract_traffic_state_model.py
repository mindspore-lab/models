from model.abstract_model import AbstractModel


class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        super().__init__(config, data_feature)

    def predict(self, batch):
        pass

    def calculate_loss(self, batch):
        pass