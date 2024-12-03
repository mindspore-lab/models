import os


def set_debug(config):
    if config.debug:
        config.dataset.batch_size = 2
        # config.dataset.GTA5.input_size = (128, 300)
        # config.dataset.Cityscapes.input_size = (128, 400)
        local_GTA5_data_dir = r"E:\data\GTA5"
        local_Cityscapes_data_dir = r"E:\data\Cityscapes"
        if os.path.exists(local_GTA5_data_dir):
            config.dataset.GTA5.data_dir = local_GTA5_data_dir
        if os.path.exists(local_Cityscapes_data_dir):
            config.dataset.Cityscapes.data_dir = local_Cityscapes_data_dir
        config.train.print_step = 1
        config.train.epoch = 1
