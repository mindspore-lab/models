class Backend(object):
    """class for inference."""

    def __init__(self, model_path):
        self.model_path = model_path

    def run(self, input_list):
        print(input_list)
