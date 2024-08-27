# [NCF Description](#contents)

NCF is a general framework for collaborative filtering of recommendations in which a neural network architecture is used to model user-item interactions. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

[Paper](https://arxiv.org/abs/1708.05031):  He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.

# [Model Architecture](#contents)

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer. [neumf_model.py](neumf_model.py) defines the architecture details.

# [Dataset](#contents)

The [Pinterest dataset](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data
) is a collection of data from the social media platform Pinterest, commonly used for research in recommendation systems, image processing, and social network analysis. These datasets typically include user interactions (such as pinning, likes, comments), images, text tags, and user relationships. It helps researchers develop and evaluate recommendation algorithms, particularly in handling images and interest classification. This dataset has already been preprocessed by the authors of Neural Collaborative Filtering.
```bash
  python data/pinterest/pinterest.py
  ```

## [Training Process](#contents)

### Training


  ```bash
  python train.py --data_path data --dataset 'pinterest'  --train_epochs 25 --batch_size 256 --output_path './output/' --checkpoint_path ncf.ckpt --device_target=Ascend > train.log 2>&1 &
  ```



## [Evaluation Process](#contents)

### Evaluation

-  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "checkpoint/ncf-125_390.ckpt".

  ```bash
  python ./eval.py --data_path data --dataset 'pinterest'  --eval_batch_size 160000  --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path ncf.ckpt/NCF-25_179.ckpt --device_target=Ascend --device_id 6 > eval.log 2>&1 &
  ```