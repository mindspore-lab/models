import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
import mindspore as ms

def gonzalez(data, k):
    n, d = data.shape
    centers = np.zeros((k, d))
    centers_t = ms.Tensor(centers, dtype=mstype.float32)
    ini_center = np.random.randint(n)
    centers_t[0, :] = data[ini_center, :]
    D = np.zeros((n, k))
    if k > n:
        print('k is too big !!')
        return
    for i in range(k - 1):

        D[:, i] = next_center(data, centers_t[i, :])  # the next center
        index = int(np.argmax(np.min(D[:, :i+1], axis=1)))
        centers_t[i + 1, :] = data[index, :]
        # indices = ms.ops.ones(data.shape[0], dtype=ms.bool_)
        # indices[index] = 0
        # data = data.index_select(axis = 0, index=indices)
        # D = np.delete(D, index, axis=0)
    return centers_t

def next_center(data, centers):
    diff_center = data - centers
    D = diff_center.norm(dim = 1)
    return D

# Tensor
data = np.random.randn(10000, 4)  # Generate some random data with 2 features
k = 5  # Set the number of clusters to 5
data_tensor = ms.Tensor(data, dtype=mstype.float32)

# run Gonzalez
result = gonzalez(data_tensor, k)
print(result)