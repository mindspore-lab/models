import os
import platform
import mindspore.dataset as ds


class SafeTrafficDataset:
    def __init__(self, data, labels, seq_length=10):
        self.data = data
        self.labels = labels
        self.seq_length = seq_length

        if len(data) != len(labels):
            raise ValueError("数据与标签长度不一致")
        if seq_length < 1:
            raise ValueError("序列长度必须大于0")

    def __getitem__(self, index):
        if index + self.seq_length > len(self.data):
            raise IndexError("索引超出数据范围")
        return (self.data[index:index + self.seq_length],
                self.labels[index + self.seq_length - 1])

    def __len__(self):
        return len(self.data) - self.seq_length + 1


def create_dataloader(data, labels, seq_length=10, batch_size=256, shuffle=True):
    """创建数据加载器

    Args:
        data (np.ndarray): 特征数据
        labels (np.ndarray): 标签数据
        seq_length (int): 序列长度
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据

    Returns:
        ds.GeneratorDataset: 数据加载器
    """
    cpu_count = os.cpu_count() or 1
    safe_num_workers = max(1, min(cpu_count - 1, 4))  # 限制最大为4

    dataset = ds.GeneratorDataset(
        source=SafeTrafficDataset(data, labels, seq_length),
        column_names=["data", "label"],
        shuffle=shuffle,
        num_parallel_workers=1 if platform.system() == 'Windows' else safe_num_workers
    )
    return dataset.batch(batch_size, drop_remainder=True)