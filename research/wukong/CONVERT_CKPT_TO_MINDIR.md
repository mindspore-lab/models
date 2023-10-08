# 将CKPT转换成MINDIR

ckpt文件中存储了模型权重， mindir不仅存储了模型权重，还存储了网络结构、输入数据类型和shape等数据。因此，需要先将ckpt加载在网络结构中，并提供输入数据，才能将ckpt转为mindir。

## 环境依赖

硬件：Ascend 910B
昇腾软件包：驱动+固件+CANN， 请前往[昇腾社区](https://www.hiascend.com/software/cann/commercial)，按照说明下载安装。
python: 3.7
Mindspore: 2.0, 请前往[MindSpore官网](https://www.mindspore.cn/install)，按照说明下载安装

## 转换方式

1. 克隆悟空画画export代码并安装依赖

```shell
git clone https://github.com/Mark-ZhouWX/minddiffusion-export-ms2.0
git checkout lite_infer_static  # 切换分支
cd ./vision/wukong-huahua
pip install -r requirements.txt  # 安装依赖
```

2. 将模型权重ckpt文件放置在models文件夹下

```shell
minddiffusion-export-ms2.0
    ├──vision
        ├── wukong-huahua  
            ├── models 
                ├── wukong-huahua-ms.ckpt  # 模型权重ckpt, 名称可自定义
```

3. 运行转换代码

```shell
python txt2img.py --ckpt_name wukong-huahua-ms.ckpt --prompt "来自深渊 风景 绘画 写实风格"  --n_samples 1 --sample_steps 15 --n_iter 1 --output_mindir_name wukong_youhua_512_512 --H 512 --W 512
```

其中，`ckpt_name`表示自定义的待转换权重名称，`prompt`为用于测试转换效果的文字提示， `output_mindir_name`为转换后的`mindir`文件名称，`H`和`W`为模型的尺寸。
转换完成后，可在`wukong-huahua`下查看生成的mindir模型(.mindir中存储了网络结构，variables文件夹中存储了模型权重，二者配套使用)和测试效果图。转换时间较长（15min左右），请耐心等待。

```shell
minddiffusion-export-ms2.0
    ├──vision
        ├── wukong-huahua
            ├── wukong_youhua_512_640_graph.mindir  # 模型结构，名称为output_mindir_name指定
            ├── wukong_youhua_512_640_variables  #  # 模型权重，名称为output_mindir_name指定
            ├──output
                ├──samples
                    ├──xxxxx.png  # 测试效果图，名称为依次递增的数字
```

## 转换原理

下面将使用伪代码描述转换的原理

1. 定义并实例化网络结构

```python
net_inference = Txt2ImgInference(model, n_samples, H, W, sample_steps, order, scale)
```

2. 再给定网络输入。悟空画画的网络输入为2个整型tensor，维度均为(batch_size, 77, 1024), 为了310P推理成功，应和310P推理batch_size保持一致，此处batch_size置为1

```python
assert prompt.shape[0] == 1 and batch_size == 1 
unconditional_condition_token = tokenize(batch_size * [""])
condition_token = tokenize(prompt)
```

执行转换

```python
import mindspore as ms
ms.export(net_inference, unconditional_condition_token, condition_token, file_name=output_mindir_name, file_format="MINDIR")
```

上述仅给出转换精简示例，具体代码实现细节请参考[悟空画画](https://github.com/Mark-ZhouWX/minddiffusion-export-ms2.0/blob/lite_infer_static/vision/wukong-huahua/txt2img.py#L143)