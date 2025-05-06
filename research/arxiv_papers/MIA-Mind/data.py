import os
import numpy as np
import pandas as pd
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import Tensor

# -------------------- ISBI2012 数据处理 --------------------
def create_isbi_dataset(img_dir="./data/isbi2012", batch_size=2, shuffle=True):
    train_img_path = os.path.join(img_dir, "train-volume.tif")
    train_mask_path = os.path.join(img_dir, "train-labels.tif")

    def load_tif_stack(path):
        stack = tiff.imread(path)
        if stack.ndim == 2:
            stack = np.expand_dims(stack, axis=0)
        return stack.astype(np.float32)

    def normalize_stack(stack):
        stack_min = np.min(stack)
        stack_max = np.max(stack)
        return (stack - stack_min) / (stack_max - stack_min + 1e-8)

    imgs = normalize_stack(load_tif_stack(train_img_path))
    masks = load_tif_stack(train_mask_path)
    masks = (masks > 127).astype(np.float32)

    data_pairs = [(img[np.newaxis, :, :], mask[np.newaxis, :, :]) for img, mask in zip(imgs, masks)]

    class ISBIDataset:
        def __init__(self, data_pairs):
            self.data = data_pairs

        def __getitem__(self, index):
            image, label = self.data[index]
            return Tensor(image), Tensor(label)

        def __len__(self):
            return len(self.data)

    dataset_generator = ISBIDataset(data_pairs)
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=shuffle)
    return dataset.batch(batch_size)

# -------------------- CIFAR-10 数据处理 --------------------
def get_cifar10_dataset(batch_size=32, resize=32, dataset_dir="./data/cifar-10-batches-bin"):
    def create_dataset(usage):
        cifar10_ds = ds.Cifar10Dataset(
            dataset_dir=dataset_dir,
            usage=usage,
            shuffle=(usage == "train")
        )

        trans = [
            vision.Resize((resize, resize), interpolation=vision.Inter.LINEAR),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            vision.HWC2CHW()
        ]

        label_trans = [transforms.TypeCast(Tensor(np.int32))]

        cifar10_ds = cifar10_ds.map(operations=trans, input_columns="image")
        cifar10_ds = cifar10_ds.map(operations=label_trans, input_columns="label")
        return cifar10_ds.batch(batch_size)

    return create_dataset("train"), create_dataset("test")

# -------------------- MachineLearningCVE 数据处理 --------------------
def load_cic_data(normal_path="./data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv", attack_path="./MachineLearningCVE/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"):
    df_normal = pd.read_csv(normal_path)
    df_attack = pd.read_csv(attack_path)

    df_normal["Label"] = 0
    df_attack["Label"] = 1

    df = pd.concat([df_normal, df_attack])
    df = df.dropna().drop_duplicates()

    df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values.astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_cic_dataset(X, y, batch_size=64):
    def generator():
        for i in range(len(X)):
            yield X[i], y[i]

    dataset = ds.GeneratorDataset(generator, column_names=["feature", "label"], shuffle=True)
    return dataset.batch(batch_size)
