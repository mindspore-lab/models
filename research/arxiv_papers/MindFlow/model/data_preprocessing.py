import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_nf_bot_iot_data(file_path):
    """加载并预处理NF-BoT-IoT数据集

    Args:
        file_path (str): 数据文件路径

    Returns:
        tuple: (features, labels, label_encoder)
    """
    df = pd.read_parquet(file_path)

    if 'Label' not in df.columns:
        raise ValueError("数据集中必须包含'Label'列")

    labels = df['Label'].copy()
    features = df.drop(columns=['Label'])

    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels).astype(np.float32)

    # 处理特征列
    for col in features.columns:
        try:
            features[col] = pd.to_numeric(features[col], errors='raise')
        except (ValueError, TypeError):
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col])

    # 标准化并转换为 float32
    features = StandardScaler().fit_transform(features).astype(np.float32)
    return features, labels, label_encoder