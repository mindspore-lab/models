# [NCF Description](#contents)

NCF is a general framework for collaborative filtering of recommendations in which a neural network architecture is used to model user-item interactions. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

[Paper](https://arxiv.org/abs/1708.05031):  He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.

# [Model Architecture](#contents)

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer. [neumf_model.py](neumf_model.py) defines the architecture details.

# [Dataset](#contents)

The [amazon_book]( https://aistudio.baidu.com/datasetdetail/109008
) are used for model training and evaluation.
This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.
This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).We use the book reviews from this dataset as training data.

Use chuli.py to process the downloaded dataset, and modify the input_file and output_file in chuli.py. Place it in your data folder.
```bash
  python data/amazon_book/Amazon_book.py
  ```

## [Training Process](#contents)

### Training


  ```bash
  python train.py --data_path data --dataset 'amazon_book'  --train_epochs 25 --batch_size 256 --output_path './output/' --checkpoint_path ncf.ckpt --device_target=Ascend > train.log 2>&1 &
  ```



## [Evaluation Process](#contents)

### Evaluation

-  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "checkpoint/ncf-125_390.ckpt".

  ```bash
  python ./eval.py --data_path data --dataset 'amazon_book'  --eval_batch_size 160000  --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path ncf.ckpt/NCF-25_179.ckpt --device_target=Ascend --device_id 6 > eval.log 2>&1 &
  ```