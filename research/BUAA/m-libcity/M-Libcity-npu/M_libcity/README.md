# M-Libcity
## 介绍
M-libcity 是一个基于华为MindSpore框架实现的开源算法库，专注于城市时空预测领域。它为MindSpore开发人员提供了统一、全面、可扩展的时空预测模型实现方案，同时为城市时空预测研究人员提供了可靠的实验工具和便捷的开发框架。M-LibCity开源算法库涵盖了与城市时空预测相关的所有必要步骤和组件，构建了完整的研究流程，使研究人员能够进行全面的对比实验。这将为研究人员在MindSpore平台上开展城市时空预测研究提供便利和强大的支持。
## 快速运行代码命令
```
cd [rootpath for project]
python test_pipeline.py
```
注意，需要去test_pipeline.py文件中更改模型和数据集

## 修改模型参数
所有的pipeline默认参数都存放在M_libcity/config文件夹下。
模型配置文件可在M_libcity/config/model文件夹下找到，该文件夹按照model的类别进行分类。
task_config.json记录了模型要加载的具体数据模块配置文件、执行模块配置文件、评估模块配置文件和模型模块配置文件，可通过task_config.json查看对应关系。
如想添加其他参数，可以在test_pipeline.py中的run_model()中通过other_args={key:value}的形式传递。

PS：所有参数的注释以及取值可从https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/config_settings.html 搜索得到。

## 数据集
所有数据集都存放在M_libcity/raw_data下。
缺少的数据集可以从网站 https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ with psw 1231 or https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing 中获得。
如想自行处理数据集，可以参照 https://github.com/LibCity/Bigscity-LibCity-Datasets 中的处理脚本。

## 调试任务实现多卡训练
调用 M_libcity/run.sh 文件
### 单卡训练
启动方式为：
```
bash run.sh 1 [task] [model_name] [dataset] or python test_pipeline.py or test_pipeline_[model_name].py
```
参数`1`表示单卡, PS: [task] 表示需要运行的任务，[model_name] 表示需要运行的模型，[dataset] 表示需要运行的数据集
### 多卡训练
启动方式为：
```
bash run.sh 2 [task] [model_name] [dataset]
```
参数`2`表示卡数为2。

