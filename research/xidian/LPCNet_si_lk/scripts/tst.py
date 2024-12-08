from numpy.typing import NDArray
import numpy as np

def func(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr * 2

print('test numpy NDArray')
a = np.array([1.0, 2.0, 3.0])
print(func(a))

print('test mindspore')
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))