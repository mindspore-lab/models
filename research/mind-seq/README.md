<div align="center" markdown>

# MindSeq

![Python 3.7](https://img.shields.io/badge/python-3.7-brightgreen.svg?style=plastic)
![MindSpore 2.0.0](https://img.shields.io/badge/MindSpore-2.0.0-618AF9.svg?style=plastic)

</div>

## 简介

MindSeq是一个基于华为[MindSpore](https://www.mindspore.cn/)开源AI框架的**时序智能计算套件**，集成了多种先进的时序建模算法，可广泛应用于时间序列领域的预测与分析任务。MindSeq套件包括Informer、NBEATS、Autoformer、FEDformer等9个领域建模&长序列SOTA算法，可以很好地适配时间序列领域各项基本任务。借助MindSpore强大的异构计算能力，MindSeq可以让时间序列模型获得较高的计算性能和扩展性，用户可以基于MindSeq快速搭建适用于各种时间序列问题的端到端解决方案。MindSeq致力于提供专业、高效的时序建模工具，以推动时间序列领域的技术和应用创新。

MindSeq支持**MindSpore 2.0🔥** 及以上版本

## 主要特性

- **使用便捷**：MindSeq可以提供给你全流程数据、模型、训练支持。支持一键完成数据准备、模型构建、模型训练、模型测试等全流程工作。MindSeq内置丰富的数据集、预训练模型，用户可以组合使用，大大简化了时序模型的开发过程。
- **算法先进**：MindSeq提供时序领域9大先进算法模型及相关参数建议，并且提供相应的预训练权重，帮助你快速选择合适的算法。
- **性能优越**：MindSeq基于MindSpore开源AI框架开发，支持CPU、GPU、Ascend等不同硬件设备，提供优越的性能保障。

## 依赖

- mindspore==2.0.0
- atari_py==0.2.9
- matplotlib==3.5.3
- numpy==1.21.6
- opencv_contrib_python_headless==4.7.0.72
- opencv_python==4.8.1.78
- opencv_python_headless==4.7.0.72
- pandas==1.3.5
- PyYAML==6.0
- Requests==2.31.0
- scikit_learn==0.20.4
- scipy==1.7.3
- tqdm==4.65.0

安装以上依赖库，只需运行

```bash
pip install -r requirements.txt
```

特别的，对于DTRD模型，在进行测试的时候需要运行以下命令来加载ROMS：
```bash
python -m atari_py.import_roms ./Roms
```

MindSpore可以通过遵循[官方指引](https://www.mindspore.cn/install/)，在不同的硬件平台上获得最优的安装体验。 为了在分布式模式下运行，您还需要安装[OpenMPI](https://www.open-mpi.org/)。


## 快速入门

MindSeq为用户提供了完整的AI模型开发全流程支持，包括数据处理、模型构建、模型训练和测试部署。MindSeq集成了丰富的公开数据集以及多种经过调优的先进算法模型，用户只需要组合相应的模型和数据集，并提供模型的参数文件，或者在执行命令时指定参数，即可快速地进行训练和预测。

下面提供具体的构建和训练过程

### 参数文件

用户可以为不同的算法模型自定义配置参数,以进行定制化的模型训练。

具体来说，configs文件夹用于存储配置文件。用户可以在对应模型的configs子文件夹下新建YAML格式的配置文件，如 `informer_train.yaml` ,在文件中指定模型的训练超参,如学习率、批大小、优化器等。在执行模型训练时，用户只需通过运行脚本或命令行接口指定该YAML配置文件即可，MindSeq将加载用户定义的配置。我们以 `informer_train.yaml` 为例对其中的一些参数做出说明

```yaml
---

model: 'Informer'
data: 'weather'
root_path: './mindseq/data/weather/'
data_path: 'weather.csv'
features: 'S'
target: 'OT'
freq: 'h'
detail_freq: 'h'
checkpoints: './checkpoints/train_ckpt'
seq_len: 96
label_len: 48
pred_len: 48

enc_in: 1
dec_in: 1
c_out: 1
d_model: 512
n_heads: 8
e_layers: 2
d_layers: 1
s_layers: '3,2,1'
d_ff: 2048
factor: 5
padding: 0
distil: True
dropout: 0.05
attn: 'prob'
embed: 'timeF'
activation: 'gelu'
output_attention: False
do_predict: False
mix: True
cols: '+'
num_workers: 0
itr: 1
train_epochs: 1
batch_size: 32
patience: 3
learning_rate: 0.0001
des: 'Informer'
loss: 'mse'
lradj: 'type1'

use_amp: False
inverse: False
seed: 42

device: "GPU"
do_train: True
ckpt_path: ''
```

其中，一些参数的说明如下：

- model：指定的模型，可选项包括 `['Informer','Autoformer','FEDformer','JAT', 'TFT', 'Nbeats', 'Nbeatsx', 'ALLOT', 'DTRD']`
- data：指定的数据集，可选项包括 `['ETTh1','ETTh2','ETTm1','ETTm2', 'weather', 'traffic', 'PEMS08', 'electricity', 'NP']`
- root_path：数据集的相对路径
- data_path：数据集名称
- checkpoints：模型训练过程中权重保存的路径
- device：硬件设备，可选项包括 `['GPU','Ascend','CPU']`
- do_train：是否进行训练，如果是True则会在数据集上进行训练，如果是False会根据指定权重的路径加载预训练权重
- ckpt_path：指定的预训练权重的加载路径，只在do_train为False的情况下有用

其他参数的说明参照 [Informer](https://github.com/zhouhaoyi/Informer2020)


### 运行脚本

设置好模型参数文件后，用户只需要简单的命令即可执行模型训练和测试，并将结果重定向到logs文件夹下的输出文件，例如

```bash
python -u train.py --model Informer --data weather -c configs/informer/informer_train.yaml > logs/Informer_ETTh1_train.log
```

其中，model和data必须指定，建议指定configs文件。MindSeq也提供了一系列运行脚本在 `scripts` 文件夹下，用户可直接执行脚本，例如

```bash
bash scripts/Informer_train.sh
```

预期运行结果如下：

```
Device: GPU
Args in experiment:
Namespace(activation='gelu', attn='prob', batch_size=32, c_out=1, checkpoint_path='', checkpoints='./checkpoints/train_ckpt', ckpt_path='', cols='+', config='./configs/informer/informer_train.yaml', config_file='', d_ff=2048, d_layers=1, d_model=512, data='ETTh1', data_name='ETTh1', data_path='ETTh1.csv', dec_in=1, des='Informer', detail_freq='h', device='GPU', device_num=None, devices='0,1,2,3', distil=True, distribute=False, do_predict=False, do_train=True, dropout=0.05, e_layers=2, embed='timeF', enc_in=1, factor=5, features='S', freq='h', gpu=0, inverse=False, itr=1, label_len=48, learning_rate=0.0001, loss='mse', lradj='type1', mix=True, model='Informer', model_name='Informer', n_heads=8, num_workers=0, output_attention=False, padding=0, patience=3, pred_len=48, pretrained=False, rank_id=None, root_path='./mindseq/data/ETT/', s_layers='3,2,1', seed=42, seq_len=96, target='OT', train_epochs=1, use_amp=False, use_gpu=True, use_multi_gpu=False)
>>>>>>>start training : Informer_ETTh1_ftS_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Informer_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 8497
val 2833
test 2833
	iters: 50, epoch: 1 | loss: 0.1454473
	speed: 0.5507s/iter; left time: 118.9614s
	iters: 100, epoch: 1 | loss: 0.2162907
	speed: 0.5033s/iter; left time: 83.5532s
	iters: 150, epoch: 1 | loss: 0.2219316
	speed: 0.5173s/iter; left time: 60.0019s
	iters: 200, epoch: 1 | loss: 0.2571105
	speed: 0.5181s/iter; left time: 34.1960s
	iters: 250, epoch: 1 | loss: 0.1243015
	speed: 0.5153s/iter; left time: 8.2446s
Epoch: 1 cost time: 137.96726202964783
Epoch: 1, Steps: 265 | Train Loss: 0.1900460 Vali Loss: 0.0942537 Test Loss: 0.0842535
Validation loss decreased (inf --> 0.094254).  Saving model ...
>>>>>>>start testing : Informer_ETTh1_ftS_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Informer_0>>>>>>>>>>>>>>>>>>>>>>>>>>
test 2833
test shape: (88, 32, 48, 1) (88, 32, 48, 1)
test shape: (2816, 48, 1) (2816, 48, 1)
mse:0.08474378287792206, mae:0.22907662391662598, rmse:0.2911078631877899

```

## 更新

- 2023/11/09

初始版本

## 贡献方式

欢迎开发者用户提issue或提交代码PR，或贡献更多的算法和模型，一起让MindSeq变得更好。

## 致谢

MindSeq是由北京航空航天大学、MindSpore团队联合开发的开源项目。 衷心感谢所有参与的研究人员和开发人员为这个项目所付出的努力。 十分感谢**北京航空航天大学大数据科学与脑机智能高精尖创新中心** 和 **[OpenI](https://openi.pcl.ac.cn/)** 平台所提供的算力资源。