# Contents

- [Contents](#contents)
- [GEHRL Description](#GEHRL-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [GEHRL Description](#contents)

We present a novel Graph Enhanced Hierarchical Reinforcement Learning (GEHRL) framework for goal-oriented learning path 
recommendation. The framework divides learning path recommendation into two parts: sub-goal selection(planning) 
and sub-goal achieving(learning item recommendation). Specifically, we employ a high-level agent as a sub-goal selector 
to select sub-goals for the low-level agent to achieve. The low-level agent in the framework is to recommend learning items 
to the learner. To make the path only contain goal-related learning items to improve the efficiency of achieving the goal, 
we develop a graph-based candidate selector to constrain the action space of the low-level agent based on the sub-goal and 
knowledge graph. We also develop test-based internal reward for low-level training so that the sparsity problem of external 
reward can be alleviated.

Graph Enhanced Hierarchical Reinforcement Learning for Goal-oriented Learning Path Recommendation

CIKM2023

# [Dataset](#contents)

- [assist15](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data)

# [Simulator](#contents)

- KESassist15

# [Environment Requirements](#contents)

- Hardware£¨CPU and GPU£©
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [MindSpore-2.0.0](https://www.mindspore.cn/install/en)
- Requirements
  - numpy
  - tqdm
  - longling
  - mindspore==2.0.0
  - gym==0.22.0
  - scikit-learn
  - genism
  
- For more information, please check the resources below£º
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

  ```shell
  # Build Simulator first
  python scripts/dataProcess.py
  # After installing MindSpore via the official website, you can start training and evaluation as follows:
  python scripts/runSim.py -s simulator -a model
  ```
# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text 
.
|-GEHRL
  |-README.md             # descriptions of GEHRL
  |-EduSim                # The simulators and agents code.
    |-__init__.py
    |-AbstractAgent.py    # The abstract class of agent.
    |-buffer.py           # The replay buffer.
    |-deep_model.py       # The basic deep models like GRU, GCN, etc.
    |-GraphEmbedding.py   # The graph embedding model of node2vec.
    |-agents              # The agents code.
      |-__init__.py
      |-AC.py             # The actor-critic agent.
      |-PPO.py            # The PPO agent.
      |-HRL.py            # The HRL agent.
    |-Envs                # The simulators code.
      |-__init__.py
      |-KES
      |-KES_ASSIST15
      |-meta
      |-shared
    |-SimOs               # The simulator os code.
      |-__init__.py
      |-SimOs.py          # The simulator os.
    |-spaces              # The action space and observation space code.
    |-utils               # Some utility functions.
  |-scripts               # data processing and model training.
    |-runSim.py           # The entry point of the training and evaluation process.
    |-dataProcess.py      # The entry point of the data processing process.
```
## [Script Parameters](#contents)

- Parameters of runSim.py

See [scripts/runSim.py](./scripts/runSim.py) for detailed parameters and explanations


[//]: # (# [Model Description]&#40;#contents&#41;)

[//]: # ()
[//]: # (## [Performance]&#40;#contents&#41;)

[//]: # ()
[//]: # (### Training Performance)

[//]: # ()
[//]: # (| Parameters          | GPU                                                                                                                        |)

[//]: # (|---------------------|----------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| Resource            | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                       |)

[//]: # (| uploaded Date       | 12/31/2023 &#40;month/day/year&#41;                                                                                                |)

[//]: # (| MindSpore Version   | 2.0.0                                                                                                                      |)

[//]: # (| Dataset             | assist15                                                                                                                   |)

[//]: # (| Simulator           | KESassist15                                                                                                                |)

[//]: # (| Training Parameters | max_steps=20, max_episode_num=15000, lr=1e-5                                                                               |)

[//]: # (| Optimizer           | Adam                                                                                                                       |)

[//]: # (| Loss Function       | Policy Gradient                                                                                                            |)

[//]: # (| Outputs             | Reward                                                                                                                     |)

[//]: # (| Results             | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper. |)

[//]: # (| Per Step Time       | 54.97 ms                                                                                                                   |)

[//]: # ()
[//]: # (### Inference Performance)

[//]: # ()
[//]: # (| Parameters        | GPU                                                                                                                         |)

[//]: # (|-------------------|-----------------------------------------------------------------------------------------------------------------------------|)

[//]: # (| Resource          | AMD Ryzen 2990WX 32-Core Processor;256G Memory;NVIDIA GeForce 2080Ti                                                        |)

[//]: # (| uploaded Date     | 01/15/2023 &#40;month/day/year&#41;                                                                                                 |)

[//]: # (| MindSpore Version | 1.9.10                                                                                                                      |)

[//]: # (| Dataset           | assist09, junyi                                                                                                             |)

[//]: # (| Simulator         | DKT, CoKT                                                                                                                   |)

[//]: # (| Outputs           | Reward                                                                                                                      |)

[//]: # (| Results           | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper.  |)

[//]: # (| Per Step Time     | 40.61 ms                                                                                                                    |)

# [Description of Random Situation](#contents)

- Simulator Training.
- Random initialization of model weights.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models)
