# Contents

- [Contents](#contents)
- [PKSD Description](#PKSD-description)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PKSD Description](#contents)
We introduce the privileged feature distillation technique and propose the Privileged Knowledge State Distillation (PKSD) framework, 
allowing the RL agent to leverage the “actual” knowledge state as privileged information in the state encoding to help tailor recommendations to meet individual needs. 
Concretely, our PKSD takes the privileged knowledge states together with the representations of the exercise log for the state representations during training. 
And through distillation, we transfer the ability to adapt to learners to a knowledge state adapter. During inference, the knowledge state adapter would serve as the estimated privileged knowledge states instead of the real one since it is not accessible. 
Considering that there are strong connections among the knowledge concepts in education, we further propose to collaborate the graph structure learning for concepts into our PKSD framework. This new approach is termed GEPKSD (Graph-Enhanced PKSD). 

Privileged Knowledge State Distillation for Reinforcement Learning-based Educational Path Recommendation

KDD2024

# [Simulator](#contents)

- KSS

# [Environment Requirements](#contents)

- Hardware: CPU and GPU
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
  
- For more information, please check the resources below��
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

  ```shell
  # Build Simulator first
  python scripts/dataProcess.py
  # After installing MindSpore via the official website, you can start training and evaluation as follows:
  python scripts/runSim.py -s simulator -a PKSD
  ```
# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text 
.
|-PKSD
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
      |-PKSD.py            # The PKSD method.
    |-Envs                # The simulators code.
      |-__init__.py
      |-KES
      |-KES_ASSIST15
      |-KSS
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

 Please check the official [homepage](https://github.com/mindspore-lab/models)
