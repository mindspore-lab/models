import os
import warnings
from os.path import join

import mindspore
import numpy as np
import random


def create_file_dirs(file_path):
    os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)

def set_seed(seed):
    print(f"Using seed: {seed}")
    mindspore.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)