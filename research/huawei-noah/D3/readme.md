# D3 

This repository contains the source code for our paper: "D3: A Methodological Exploration of Domain Division, Modeling, and Balance in Multi-Domain Recommendations".

## Environment

```
torch
scikit-learn
pandas
tqdm
```

## BibTex

```
@inproceedings{jia2024d3,
    title={D3: A Methodological Exploration of Domain Division, Modeling, and Balance in Multi-Domain Recommendations},
    author={Jia, Pengyue and Wang, Yichao and Lin, Shanru and Li, Xiaopeng and Zhao, Xiangyu and Guo, Huifeng and Tang, Ruiming},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={38},
    number={8},
    pages={8553--8561},
    year={2024}
}
```

## Usage

This code is used for aliccp dataset, for the other datasets, please modify the corresponding slot_id index in `run.py` and `D3.py`.

run the `run.py` file. This can be done by navigating to the project directory and executing the following command in the terminal.

```bash
python run.py
```

Ensure that all the dependencies are installed and the dataset is putted under path `./data/` before running the project.

