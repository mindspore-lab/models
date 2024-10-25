# Influence Function based Recommendation Quality Exploration
Would You Like Your Data to Be Trained? A User Controllable Recommendation Framework

# Contents

- [Influence Function based Recommendation Quality Exploration](#influence-function-based-recommendation-quality-exploration)
- [Contents](#contents)
- [IFRQE Description](#ifrqe-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
- [Model Description](#model-description)
  - [Performance](#performance)
    - [Evaluation Performance](#evaluation-performance)
    - [Evaluation Performance](#evaluation-performance-1)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [IFRQE Description](#contents)

IFRQE is a general framework, where the users can explicitly indicate their disclosing willingness, and the model needs to trade-off the recommendation quality and user willingness. IFRQE first finds the best strategy that balances the recommendation quality and user willingness. Then the user interactions are disclosed according to the optimal strategy. At last, the recommender model is trained based on the disclosed interactions.

AAAI 2024

# [Model Architecture](#contents)

IFRQE is a framework based on the NCF model. Firstly, the NCF model is trained normally. Then, by running `influence_function.py`, the influence of each data on the results is calculated using the influence function, and the strategy that maximizes user profits is determined based on user exposure preferences. The data is then masked to generate the `masked_ml-1m` dataset. The NCF model is then retrained using the `masked_ml-1m` dataset, and the performance of the two models is evaluated.

# [Dataset](#contents)

The [MovieLens datasets](http://files.grouplens.org/datasets/movielens/) are used for model training and evaluation. The
ml-1m dataset contains 1,000,209 anonymous ratings of approximately 3,706 movies made by 6,040 users who joined MovieLens in 2000. All ratings are contained in the file "ratings.dat" without header row, and are in the following format:

```cpp
  UserID::MovieID::Rating::Timestamp
```

- UserIDs range between 1 and 6040.
- MovieIDs range between 1 and 3952.
- Ratings are made on a 5-star scale (whole-star ratings only).



# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [MindSpore-2.0.0](https://www.mindspore.cn/install/en)
- Requirements
  - numpy
  - tqdm
  - pandas
  - argparse
  - skit-learn
  - mindspore==2.0.0

- For more information, please check the resources below:
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
  
# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
#run data process
bash scripts/run_download_dataset.sh

# run calculating influence function example on GPU
bash scripts/run_preprocess.sh

# run training example on GPU
bash scripts/run_train_gpu.sh

# run evaluation example on GPU
bash scripts/run_eval_gpu.sh
```



# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── ModelZoo_IFRQE_ME
    ├── README.md                          // descriptions about IFRQE
    ├── scripts
    │   ├──run_train_gpu.sh               // shell script for train on GPU
    |   ├──run_preprocess.sh                // shell script for calculating influence function
    │   ├──run_eval_gpu.sh                 // shell script for evaluation on GPU
    │   ├──run_download_dataset.sh         // shell script for dataget and process
    ├── src
    │   ├──dataset.py                      // creating dataset
    │   ├──ncf.py   // ncf architecture
    |   ├──influence_function.py                       // calculate influence function
    │   ├──config.py                       // parameter analysis
    │   ├──device_adapter.py               // device adapter
    │   ├──local_adapter.py                // local adapter
    │   ├──moxing_adapter.py               // moxing adapter
    │   ├──movielens.py                    // data download file
    │   ├──callbacks.py                    // model loss and eval callback file
    │   ├──constants.py                    // the constants of model
    │   ├──metrics.py                      // the file for auc compute
    │   ├──stat_utils.py                   // the file for data process functions
    ├── default_config.yaml    // parameter configuration
    ├── preprocess.py               // training script
    ├── train.py               // training script
    ├── eval.py                //  evaluation script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for IFRQE, ml-1m dataset

  ```text
  * `--data_path`: This should be set to the same directory given to the data_download data_dir argument.
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is ml-1m.
  * `--train_epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--num_neg`: The Number of negative instances to pair with a positive instance.
  * `--layers`： The sizes of hidden layers for MLP.
  * `--num_factors`：The Embedding size of MF model.
  * `--output_path`：The location of the output file.
  * `--eval_file_name` : Eval output file.
  * `--masked` : Whether IFRQE or the original model is evaluated.
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters          | GPU                                                    | CPU                                                    |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| Model Version       | IFRQE                                                  | IFRQE                                                  |
| Resource            | NVIDIA Corporation GV100 [TITAN V] (rev a1)            | CPU AMD R75800H,3.2GHz,6cores;Memory 16G; 0S Windows11 |
| uploaded Date       | 03/28/2023 (month/day/year)                            | 03/27/2023 (month/day/year)                            |
| MindSpore Version   | 1.9.0                                                  | 1.9.0                                                  |
| Dataset             | ml-1m                                                  | ml-1m                                                  |
| Training Parameters | epoch=25, steps=19418, batch_size = 256, lr=0.00382059 | epoch=25, steps=19418, batch_size = 256, lr=0.001      |
| Optimizer           | GradOperation                                          | GradOperation                                          |
| Loss Function       | Softmax Cross Entropy                                  | Softmax Cross Entropy                                  |
| outputs             | probability                                            | probability                                            |
| Speed               | 1pc: 2.5 ms/step                                       | 1pc: 4.2 ms/step                                       |
| Total time          | 1pc: 25 mins                                           | 1pc: 29 mins                                           |





### Evaluation Performance

| Parameters        | CPU                         | CPU                        |
| ----------------- | --------------------------- | -------------------------- |
| Model Version     | IFRQE                       | NCF                        |
| Resource          | AAMD R75800H; OS Windows11  | AMD R75800H; OS Windows11  |
| Uploaded Date     | 3/287/2023 (month/day/year) | 3/27/2023 (month/day/year) |
| MindSpore Version | 1.9.0                       | 1.9.0                      |
| Dataset           | ml-1m                       | ml-1m                      |
| batch_size        | 256                         | 256                        |
| Accuracy          | Reward: -3.18               | Reward: -4.96              |


# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
