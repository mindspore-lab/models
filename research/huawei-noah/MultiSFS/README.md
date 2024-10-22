# MultiSFS

Official mindspore implementation for the paper "Single-shot Feature Selection for Multi-task Recommendations" in SIGIR'23. 


BibTex for Citing our paper:

```
@inproceedings{wang2023single,
  title={Single-shot Feature Selection for Multi-task Recommendations},
  author={Wang, Yejing and Du, Zhaocheng and Zhao, Xiangyu and Chen, Bo and Guo, Huifeng and Tang, Ruiming and Dong, Zhenhua},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={341--351},
  year={2023}
}
```

## Contents

Backbone Multi-task recommendation  models (AITM, PLE) are provided in *models/*. 

The body of MultiSFS is in *pruner.py*, training process is in *main.py*.

## How to run 
Here is an example code to run MultiSFS for PLE on KuaiRand dataset, dropping half of feature fields.
```Shell
python main.py --dataset_name KuaiRand --model_name ple --cr 0.5
```
Other modifications on hyperparameters can refer to *main.py*.

