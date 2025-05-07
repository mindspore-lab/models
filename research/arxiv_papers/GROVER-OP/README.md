
# 基于MindSpore的精确多目标量子搜索算法的优化
# The optimization of exact multi-target quantum search algorithm based on MindSpore

Grover’s search algorithm has attracted great attention due to its quadratic speedup over classical algorithms in unsorted database search problems. However, Grover’s algorithm is inefficient in multi-target search problems, except in the case of 1/4 of the data in the database satisfying the search conditions. Long presented a modified version of Grover’s search algorithm by introducing a phase-matching condition, which can search for the target state with zero theoretical failure rate. In this work, we present an optimized exact multi-target search algorithm based on the modified Grover’s algorithm, by transforming the canonical diffusion operator to a more efficient diffusion operator, which can solve the multi-target search problem with a 100% success rate while requiring fewer gate counts and shallower circuit depth. After that, the optimized multi-target algorithm for four different items, including 2-qubit with 2 targets, 5-qubit with 2 targets, 5-qubit with 4 targets, and 6-qubit with 3 targets, are implemented on a quantum computing framework MindQuantum. The experimental results show that, compared with Grover’s algorithm and the modified Grover’s algorithm, the proposed algorithm can reduce the quantum gate count by at least 21.1% and the depth of the quantum circuit by at least 11.4% and maintain a 100% success probability.

Keywords: Quantum computing , Grover's algorithm , Optimized multi-target search algorithm

Folder: GROVER-Grover's algorithm, OP-Optimized
## Test Environment:

```python
- Mindquantum 0.9.11
- scipy 1.9.3
- numpy 1.24.2
- Python 3.9.11
- OS Linux x86_64
- CPU Max Thread 64
```


## Installation

Install Mindquantum with pip.
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp37-cp37m-win_amd64.whl
```
Verify if the installation was successful
```bash
python -c 'import mindquantum'.
```
    
## Acknowledgements

 - [Thank you to the MindSpore community for your support!](https://www.mindspore.cn/)


