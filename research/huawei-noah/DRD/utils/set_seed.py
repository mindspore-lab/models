# import torch
import mindspore
import numpy as np
import random

def setup_seed(seed):
     mindspore.set_seed(seed)
     # torch.manual_seed(seed)
     # torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # torch.backends.cudnn.deterministic = True

if __name__ =="__main__":
    setup_seed(42)