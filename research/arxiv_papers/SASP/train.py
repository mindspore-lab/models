import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.nn as nn
import mindspore
from mindspore import Tensor, ops
import mindspore as ms
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
print("当前设备：", context.get_context("device_target"))


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cub_root', type=str, required=True,
                        help='Bird/CUB200-2011 (必须包含images.txt等文件)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=8, help='Number of attention head.')
    return parser.parse_args()


import os
from PIL import Image


class Dataset:
    def __init__(self, root, phase, transform=None):
        """
        root: CUB_200_2011 根目录
        phase: 'train' or 'test'
        transform: PIL->Tensor 等操作
        """
        self.root = root
        self.phase = phase
        self.transform = transform

        # 1. 读取 images.txt
        images_file = os.path.join(root, "images.txt")
        # mapping: img_id(int) -> relative_path(str)
        self.id2path = {}
        with open(images_file) as f:
            for line in f:
                img_id, rel_path = line.strip().split()
                self.id2path[int(img_id)] = rel_path

        # 2. 读取 train_test_split.txt
        split_file = os.path.join(root, "train_test_split.txt")
        self.train_ids = set()
        with open(split_file) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                img_id, is_train = int(img_id), int(is_train)
                if is_train == 1:
                    self.train_ids.add(img_id)

        # 3. 读取 labels（类别索引）
        labels_file = os.path.join(root, "image_class_labels.txt")
        self.id2label = {}
        with open(labels_file) as f:
            for line in f:
                img_id, cls = line.strip().split()
                self.id2label[int(img_id)] = int(cls) - 1  # 0-based

        print(f"[Dataset init] 总图片数: {len(self.id2path)}"
              f", train 样本数: {len(self.train_ids)}")

        # 4. 根据 phase 过滤出要用的 img_ids
        if phase == 'train':
            self.selected = sorted(self.train_ids)
        else:
            # 测试集：所有不在 train_ids 里的
            all_ids = set(self.id2path.keys())
            self.selected = sorted(all_ids - self.train_ids)

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        try:
            img_id = self.selected[idx]
            img_path = os.path.join(self.root, "images", self.id2path[img_id])
            label = self.id2label[img_id]

            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

        except Exception as e:
            print(f"[__getitem__] 错误: idx={idx}, img_id={img_id}, path={img_path}, error={e}")
            raise

    def generator(self):
        for i in range(len(self)):
            try:
                yield self.__getitem__(i)
            except Exception as e:
                print(f"[generator] 第 {i} 个样本处理失败: {e}")
                raise


args = parse_args()

gen_ds = ds.GeneratorDataset(
    Dataset(args.cub_root, 'train', transform=None).generator(),
    column_names=["image", "label"],
    shuffle=True
)
print(">>> 1) gen_ds size:", gen_ds.get_dataset_size())


def safe_preprocess(image, label):
    # 最关键：第一行就打印，确认 map 真跑了
    print(">>> safe_preprocess called")
    img = trans(image)
    lbl = type_cast_op(label)
    return img, lbl


# 一定要把结果重新赋给 train_ds
train_ds = gen_ds.map(
    operations=safe_preprocess,
    input_columns=["image", "label"],
    output_columns=["image", "label"],
    num_parallel_workers=1,
    python_multiprocessing=False
)

it = train_ds.create_tuple_iterator()
for i in range(3):
    try:
        data, label = next(it)
        print(f"[iter test] Batch {i} OK")
    except Exception as e:
        print(f"[iter test] 第 {i} 批失败: {e}")
        break

# 先 project
train_ds = train_ds.project(["image", "label"])
# 再 batch
train_ds = train_ds.batch(args.batch_size, drop_remainder=True)

# 再迭代一次确认
it = train_ds.create_tuple_iterator()
try:
    batch_data, batch_label = next(it)
    print(f"[batch test] OK: batch_data.shape={batch_data.shape}")
except Exception as e:
    print(f"[batch test] 失败: {e}")

from mindspore import save_checkpoint
from mindspore import ops


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

    if accuracy > 0.84 and accuracy > best_accuracy:
        best_accuracy = accuracy
        # Create dynamic save path using variables (e.g., using accuracy or epoch)
        dynamic_save_path = f"./checkpoints3/best_model_accuracy_{100 * accuracy:.2f}.ckpt"
        save_checkpoint(model, dynamic_save_path)
        print(f"Model saved with accuracy: {accuracy * 100:.2f}% at {dynamic_save_path}")

    print(
        f"Test: \n Accuracy: {(100 * accuracy):>0.2f}%, Avg loss: {test_loss:>8f},best_Accuracy: {(100 * best_accuracy):>0.2f}%, \n")


# 训练集增强（使用更大尺寸的随机裁剪）
trans = transforms.Compose([
    vision.Resize(256),
    vision.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 更灵活的裁剪
    vision.RandomHorizontalFlip(0.5),
    vision.RandomRotation(30),
    vision.ToTensor(),
    vision.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
    vision.RandomErasing(prob=0.5, scale=(0.02, 0.2))  # 更高的擦除概率
])

# 验证集处理
trans_val = transforms.Compose([
    vision.Resize(256),
    vision.CenterCrop(224),  # 中心裁剪保证一致性
    vision.ToTensor(),
    vision.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
])

# 类型转换
type_cast_op = transforms.TypeCast(ms.int32)

# 只做一次 map
train_dataset = gen_ds.map(
    operations=safe_preprocess,
    input_columns=["image", "label"],
    output_columns=["image", "label"],
    num_parallel_workers=1,
    python_multiprocessing=False,  # 关键：禁用多进程
)

eval_dataset = eval_dataset.map(operations=trans_val, input_columns="image")
eval_dataset = eval_dataset.map(operations=type_cast_op, input_columns="label")

# 批次处理
train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
eval_dataset = eval_dataset.batch(args.batch_size, drop_remainder=False)

it = train_dataset.create_tuple_iterator()
for i in range(3):
    try:
        data, label = next(it)
        print(f"[iter test] 第 {i} 批 OK: data.shape={data.shape}, label.shape={label.shape}")
    except Exception as e:
        print(f"[iter test] 第 {i} 批失败: {e}")
        break
train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
it = train_dataset.create_tuple_iterator()
try:
    batch_data, batch_label = next(it)
    print(f"[batch test] data batch shape: {batch_data.shape}, label batch shape: {batch_label.shape}")
except Exception as e:
    print(f"[batch test] 失败: {e}")

config.steps_per_epoch = train_dataset.get_dataset_size()
model = FineB()
loss_fn = nn.CrossEntropyLoss()
polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01, decay_steps=4, power=0.5)
optimizer = nn.Momentum(model.trainable_params(), learning_rate=polynomial_decay_lr, momentum=0.9, weight_decay=0.0)
it = train_dataset.create_tuple_iterator()
for i in range(5):
    try:
        data, label = next(it)
        print(f"Batch {i} OK, data.shape={data.shape}, label.shape={label.shape}")
    except Exception as e:
        print(f"[迭代测试] 第 {i} 个 batch 失败: {e}")
        break

