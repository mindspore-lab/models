class DataIO:
    def set_input_model_shape(self, **kwargs):
        pass

    def set_output_model_shape(self, **kwargs):
        pass

    def set_scale(self, scale):
        pass

    def preprocess(self, input_data):
        pass

    def postprocess(self, input_data):
        pass

    def save_result(self, output_file, output_data):
        pass
