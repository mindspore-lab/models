import numpy as np
import pandas as pd

        
class KuaiRand():
    def __init__(self, dataset_path, task_t):
        data = pd.read_csv(dataset_path).to_numpy()[:, 2:]
        self.categorical_data = data[:, :41].astype(np.int)
        if task_t == 0: 
            self.labels = data[:, [41,47]].astype(np.float32)  #'is_click', 'long_view', 'is_like','is_follow', 'is_comment', 'is_forward', 'is_hate'
        elif task_t == 1:
            self.labels = data[:, [41,43]].astype(np.float32)
        elif task_t == 2:
            self.labels = data[:, [43,45]].astype(np.float32)
        elif task_t == 3:
            self.labels = data[:, [41,43,45]].astype(np.float32)
        elif task_t == 4:
            self.labels = data[:, [41,43,45,46]].astype(np.float32)
        else:
            return 0
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.labels[index]

class AliCCP():
    def __init__(self, dataset_path, task_num=2):
        data = pd.read_csv(dataset_path).to_numpy()
        self.categorical_data = data[:, 2:].astype(np.int)
        self.labels = data[:, :2].astype(np.float32)  #'is_click', 'long_view', 'is_like','is_follow', 'is_comment', 'is_forward', 'is_hate'
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.labels[index]

class Ten():
    def __init__(self, dataset_path, task_num=2):
        data = pd.read_csv(dataset_path).to_numpy()
        self.categorical_data = data[:, :-2].astype(np.int)
        self.labels = data[:, -2:].astype(np.float32)  #'is_click', 'long_view', 'is_like','is_follow', 'is_comment', 'is_forward', 'is_hate'
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.labels[index]
