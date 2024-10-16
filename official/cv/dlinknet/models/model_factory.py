from models.dlinknet import DinkNet34, DinkNet50

def create_model(
    model_name: str,
):
    if model_name == 'dinknet34':
        network = DinkNet34(use_backbone=True)
    else:
        network = DinkNet50(use_backbone=True)
    return network
