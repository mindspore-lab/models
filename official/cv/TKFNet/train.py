from mindspore.dataset import GeneratorDataset
from config import config
from model import OURS
import os
import warnings
import argparse
from PIL import Image
import pandas as pd
import mindspore.dataset.transforms as transforms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore as ms
from mindspore import context
import mindspore
from mindspore import save_checkpoint
from mindspore import ops
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

print("当前设备：", context.get_context("device_target"))




def warn(*args, **kwargs):
    pass


warnings.warn = warn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=8, help='Number of attention head.')
    return parser.parse_args()


class DataSet:
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        # 读取标签文件
        df = pd.read_csv(os.path.join(self.raf_path, "./datasets/RAF_224/list_patition_label.txt"),
                         sep=' ', header=None,
                         names=['name', 'label'])

        # 根据 phase 选择数据集
        if phase == 'train':  # 训练阶段加载训练集
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values -1

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0] + ".jpg"
            path = os.path.join(self.raf_path, "./datasets/RAF_224", f)
            if not os.path.exists(path):
                print(f"⚠️ 图像路径不存在: {path}")
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 图像读取失败: {path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            label = -1
            return image, label

        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def generator(self):
        """生成器函数，用于返回 (image, label)"""
        for idx in range(len(self.file_paths)):
            image, label = self.__getitem__(idx)
            yield image, label

def create_dataset(raf_path, phase, transform=None):
    dataset = DataSet(raf_path, phase, transform)
    generator = dataset.generator()

    return ds.GeneratorDataset(generator, column_names=["image", "label"], num_parallel_workers=1 ,shuffle=True)










def train_loop(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")



def test_loop(model, dataset, loss_fn):
    best_accuracy = 0.0
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()


        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    accuracy =correct
    print(f"Test: \n Accuracy: {(100 *correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    if accuracy > 0.84 and accuracy > best_accuracy:
        best_accuracy = accuracy
        dynamic_save_path = f"./checkpoints3/best_model_accuracy_{100 *accuracy:.2f}.ckpt"
        save_checkpoint(model, dynamic_save_path)
        print(f"Model saved with accuracy: {accuracy * 100:.2f}% at {dynamic_save_path}")

    print (f"Test: \n Accuracy: {(100 * accuracy):>0.2f}%, Avg loss: {test_loss:>8f},best_Accuracy: {(100 * best_accuracy):>0.2f}%, \n")




trans = transforms.Compose([
    vision.RandomHorizontalFlip(0.5),
    vision.RandomRotation(20),
    vision.RandomCrop(224, padding=32),
    vision.ToTensor(),
    vision.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
    vision.RandomErasing(scale=(0.02, 0.25))])
trans_val = transforms.Compose([
    vision.ToTensor(),
    vision.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225] ,is_hwc=False),
])
train_data = DataSet("", phase='train')
eval_data = DataSet("", phase='test')

train_dataset = GeneratorDataset(train_data, column_names=["image", "label"])
eval_dataset = GeneratorDataset(eval_data, column_names=["image", "label"])

type_cast_op = transforms.TypeCast(ms.int32)

train_dataset = train_dataset.map(operations=trans, input_columns="image", num_parallel_workers=1)
train_dataset = train_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)

eval_dataset = eval_dataset.map(operations=trans_val, input_columns="image", num_parallel_workers=1)
eval_dataset = eval_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
train_dataset = train_dataset.batch(config.batch_size, drop_remainder=False)
eval_dataset = eval_dataset.batch(config.batch_size, drop_remainder=False)
config.steps_per_epoch = train_dataset.get_dataset_size()

model = OURS()
loss_fn = nn.CrossEntropyLoss()




polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01, decay_steps=4, power=0.5)
optimizer = nn.Momentum(model.trainable_params(), learning_rate=polynomial_decay_lr, momentum=0.9, weight_decay=0.0)





epochs = 60
for t in range(epochs):
    print(f"Epoch { t +1}\n-------------------------------")
    train_loop(model, train_dataset, loss_fn, optimizer)
    test_loop(model, eval_dataset, loss_fn)
print("Done!")
