import os
import math
import random
import numpy as np
import mindspore as ms
from mindspore import dataset as ds
from mindspore.experimental import optim
from module.utils import cosine_schedule_with_warmup
from eval import eval
from config import Config
from model import CaptionModel
from read_file import build_data

def set_seed(seed):
    random.seed(seed)  # 配置Python random库的随机种子
    np.random.seed(seed)  # 配置Numpy库的随机种子
    ms.set_seed(seed)  # 配置MindSpore库的随机种子

def train(config):

    # 加载模型
    model = CaptionModel(config)
    print(model)

    # 读取数据
    print("读取数据")
    ds.config.set_auto_offload(True)
    ds.config.set_enable_autotune(True)
    column_names = ['img', 'caption', 'label', 'img_id']
    train_dict = build_data(config)
    train_data = ds.GeneratorDataset(source = train_dict, column_names = column_names, shuffle = True)
    train_data = train_data.batch(config.batch_size)
    train_data = train_data.create_dict_iterator()

    configVal = Config(TrainOrVal = 'val')
    val_dict = build_data(configVal)
    val_data = ds.GeneratorDataset(source = val_dict, column_names = column_names, shuffle = False)
    val_data = val_data.batch(config.batch_size)
    val_data = val_data.create_dict_iterator()

    configTest = Config(TrainOrVal = 'test')
    test_dict = build_data(configTest)
    test_data = ds.GeneratorDataset(source = test_dict, column_names = column_names, shuffle = False)
    test_data = test_data.batch(config.batch_size)
    test_data = test_data.create_dict_iterator()

    print("train data is: ", len(train_dict))
    print("val data is: ", len(val_dict))
    print("test data is: ", len(test_dict))
    print("读取数据结束")

    all_steps = math.ceil(len(train_dict) / config.batch_size)
    optimizer = optim.Adam(params=model.trainable_params(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, config.epoch * all_steps // 10, config.epoch * all_steps)

    def forward_fn(batch):
        img = batch['img']
        caption = batch['caption']
        label = batch['label']
        loss = model(img, caption, label)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # 开始训练
    for epoch in range(config.epoch):
        print(scheduler.get_last_lr())
        model.set_train(True)
        for i, batch in enumerate(train_data):
            loss, grads = grad_fn(batch)
            optimizer(grads)
            scheduler.step()
            if i % 100 == 0:
                print('i/batch: {}/{} | epoch/epochs: {}/{} | loss: {}'.format(i, all_steps, epoch, config.epoch, loss.item()))

        ms.save_checkpoint(model, os.path.join(config.model_save_path, 'epoch_{}.ckpt'.format(epoch)))
        print("test:", end = ' ')
        with ms._no_grad():
            eval(configVal, model, val_data, val_dict)

    with ms._no_grad():
        eval(configTest, model, test_data, test_dict)

if __name__ == '__main__':
    ms.set_context(mode = ms.PYNATIVE_MODE, device_target = 'GPU')
    set_seed(Config().seed)
    config = Config()
    train(config)