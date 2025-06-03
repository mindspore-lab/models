from mindspore.dataset import GeneratorDataset
import numpy as np
import mindspore as ms
from mindspore.train import Model
from mindspore import nn
from mindspore import Profiler
from moxing_adapter import moxing_wrapper
from config import config
from env import init_env
from mindspore import dtype as mstype
import pandas as pd
from PIL import Image
import mindspore.dataset as ds
import matplotlib.pyplot as plt
import warnings
import argparse
from PIL import Image
import warnings
from tqdm import tqdm
import argparse
import mindspore.nn as nn
from mindvision.classification.models import resnet50
from mindspore import load_checkpoint, load_param_into_net
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore
from mindspore import Tensor, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.experimental import optim
import mindspore.dataset.vision as vision
import mindspore.ops as ops
import os
import argparse
import warnings
from mindspore import context, nn, ops, save_checkpoint
from model import FineB
from mindspore.nn import WithLossCell, TrainOneStepCell
import traceback
from mindspore import Tensor
from download import download
import matplotlib.pyplot as plt
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import initializer
from mindspore import load_checkpoint
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
print("当前设备：", context.get_context("device_target"))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
warnings.warn = lambda *a, **k: None
parser = argparse.ArgumentParser()
parser.add_argument('--cub_root', type=str, required=True, default='/tmp/code/Bird/CUB200-2011')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=60)
args = parser.parse_args()


class Dataset:
    def __init__(self, root, phase, transform=None):
        self.root, self.phase, self.transform = root, phase, transform
        # load meta files…
        self.id2path = {}
        with open(os.path.join(root, "images.txt")) as f:
            for l in f:
                i, p = l.strip().split()
                self.id2path[int(i)] = p
        self.train_ids = set()
        with open(os.path.join(root, "train_test_split.txt")) as f:
            for l in f:
                i, fl = map(int, l.split())
                if fl == 1: self.train_ids.add(i)
        self.id2label = {}
        with open(os.path.join(root, "image_class_labels.txt")) as f:
            for l in f:
                i, c = l.split()
                self.id2label[int(i)] = int(c) - 1

        print(f"[Dataset init] phase={phase}, total={len(self.id2path)}, train_count={len(self.train_ids)}")
        if phase == 'train':
            self.selected = sorted(self.train_ids)
        else:
            all_ids = set(self.id2path)
            self.selected = sorted(all_ids - self.train_ids)

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        i = self.selected[idx]
        path = os.path.join(self.root, "images", self.id2path[i])
        img = Image.open(path).convert('RGB')
        # if self.transform: img = self.transform(img)
        if self.transform:
            img = self.transform(img)
            if isinstance(img, (tuple, list)):
                img = img[0]  # ✅ 解包成真正的 tensor

        return img, self.id2label[i]

    def generator(self):
        for i in range(len(self)):
            yield self[i]


from mindspore import save_checkpoint


def train_loop(model, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)

        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
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
    accuracy = correct
    print(f"Test: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    if accuracy > 0.79 and accuracy > best_accuracy:
        best_accuracy = accuracy
        # Create dynamic save path using variables (e.g., using accuracy or epoch)
        dynamic_save_path = f"./checkpoints/best_model_accuracy_{100 * accuracy:.2f}.ckpt"
        save_checkpoint(model, dynamic_save_path)
        print(f"Model saved with accuracy: {accuracy * 100:.2f}% at {dynamic_save_path}")

    print(
        f"Test: \n Accuracy: {(100 * accuracy):>0.2f}%, Avg loss: {test_loss:>8f},best_Accuracy: {(100 * best_accuracy):>0.2f}%, \n")


trans = transforms.Compose([
    vision.Resize((256, 256)),
    vision.RandomHorizontalFlip(0.5),
    vision.RandomRotation(20),
    vision.RandomCrop(224, padding=32),
    vision.ToTensor(),
    vision.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
    vision.RandomErasing(scale=(0.02, 0.25))])
trans_val = transforms.Compose([
    vision.Resize((256, 256)),
    vision.RandomCrop(224, padding=32),
    vision.ToTensor(),
    vision.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225], is_hwc=False),
])


def safe_preprocess(img, lbl):
    try:
        print(f">>> {phase} safe_preprocess", end="\r")  # 只需看到“在跑”即可
        img2 = transform(img) if phase == 'train' else trans_val(img)
        lbl2 = type_cast_op(lbl)
        return img2, lbl2
    except Exception as e:
        print(f"[{phase}] ERROR in safe_preprocess:", e)
        raise


def safe_preprocess_eval(image, label):
    try:
        # 你的预处理逻辑
        img2 = trans_val(image)  # 用验证集的 transform
        lbl2 = type_cast_op(label)
        return img2, lbl2
    except Exception as e:
        print(f"[safe_preprocess_eval] Error processing sample: {e}")
        # 打印完整的 traceback
        traceback.print_exc()
        # 同时打印一下输入的类型／形状
        try:
            print("  image type:", type(image), getattr(image, "size", None))
            print("  label value:", label)
        except:
            pass
        raise


eval_data = Dataset(args.cub_root, 'eval', transform=trans_val)
eval_dataset = GeneratorDataset(eval_data, column_names=["image", "label"], shuffle=True)

train_data = Dataset(args.cub_root, 'train', transform=trans)
train_dataset = GeneratorDataset(train_data, column_names=["image", "label"], shuffle=True)

type_cast_op = transforms.TypeCast(ms.int32)

train_dataset = train_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
eval_dataset = eval_dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)

eval_dataset = eval_dataset.batch(config.batch_size, drop_remainder=False)
train_dataset = train_dataset.batch(config.batch_size, drop_remainder=False)
config.steps_per_epoch = train_dataset.get_dataset_size()

model = FineB()
loss_fn = nn.CrossEntropyLoss()

polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01, decay_steps=4, power=0.5)
optimizer = nn.Momentum(model.trainable_params(), learning_rate=polynomial_decay_lr, momentum=0.9, weight_decay=0.0)

epochs = 70
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(model, train_dataset, loss_fn, optimizer)
    test_loop(model, eval_dataset, loss_fn)
print("Done!")
