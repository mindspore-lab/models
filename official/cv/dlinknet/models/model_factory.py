from models.dlinknet import DLinkNet34, DLinkNet50

def create_model(
    model_name: str,
):
    if model_name == 'dlinknet34':
        network = DLinkNet34(use_backbone=True)
    else:
        network = DLinkNet50(use_backbone=True)
    return network
