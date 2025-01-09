# Noise-Mitigated Variational Quantum Eigensolver (MPS-VQE)

This repository contains the implementation of the **Noise-Mitigated Variational Quantum Eigensolver (MPS-VQE)** algorithm, as described in the paper:

**"Noise-Mitigated Variational Quantum Eigensolver with Pre-training and Zero-Noise Extrapolation"**  
Authors: Wanqi Sun, Chenghua Duan, Jungang Xu  
Link: *[arXiv:2501.01646v1 [quant-ph]](https://arxiv.org/abs/2501.01646)*  

The MPS-VQE algorithm is designed to compute molecular ground state energies on noisy intermediate-scale quantum (NISQ) devices. It incorporates **Matrix Product States (MPS)** for circuit design, **pre-training** for parameter initialization, and **zero-noise extrapolation (ZNE)** combined with neural networks for noise mitigation.

---

## Experimental Results

The experiments were conducted using the quantum computing toolkit [**MindQuantum**](https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.9/parameterized_quantum_circuit.html) within the [**MindSpore**](https://www.mindspore.cn/) framework. 
The following table summarizes the performance of the MPS-VQE algorithm compared to other methods for the H₄ molecule:

| Model          | Noiseless (Hartree) | Noisy (Hartree) |
|----------------|---------------------|-----------------|
| MPS-VQE        | -2.1609             | -2.1490         |
| HE-VQE         | -2.1723             | -1.6726         |
| Qubit UCC      | -2.1476             | -0.6916         |
| SE Ansatz      | -2.1200             | -1.5781         |
| UCCSD          | -2.1615             | -0.5293         |
| FCI Benchmark  | -2.1664             | -2.1664         |

The molecular configuration used in this experiment is:

```
['H 0 0 1 Å', 'H 0 0 2 Å', 'H 0 0 3 Å', 'H 0 0 4 Å']
```

---

## Requirements

To run the code, ensure you have the following dependencies installed:

- Python 3.8
- [MindQuantum 0.9.11](https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.9/index.html)
- NumPy
- SciPy

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Experiment

The main script for running the MPS-VQE experiment is `VQE_with_mps_circ.py`. To execute the experiment for the molecule:

```bash
python VQE_with_mps_circ.py
```

To compare the performance of different VQE circuit designs, use the script `VQE_with_different_circs.py`. Run the following command:

```bash
python VQE_with_different_circs.py
```

## Project Structure

```
mps-vqe/  
├── VQE_with_mps_circ.py       # Main script for running MPS-VQE experiments  
├── VQE_with_different_circs.py # Script for comparing different VQE circuit designs  
├── mps_reference_code.py      # Reference implementation of MPS  
├── noise_model.py             # Noise model definitions for simulations  
├── simulator.py               # Quantum simulator for running experiments  
├── utils.py                   # Utility functions for data processing and visualization  
├── data_mol/                  # Input molecular data for experiments  
│   ├── mol.csv                # Molecular data for H_4 
│   ├── mol_H2O.csv            # Molecular data for H2O  
│   ├── mol_H4.csv             # Molecular data for H_4  
│   ├── mol_LiH.csv            # Molecular data for LiH  
│   └── mol_test.csv           # Test molecular data for debugging  
├── results/                   # Output results and visualizations  
├── requirements.txt           # Python dependencies  
└── README.md                  # Project documentation
```


---


