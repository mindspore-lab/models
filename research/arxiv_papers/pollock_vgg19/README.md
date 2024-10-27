# 基于VGG19的波洛克风格迁徙绘画分形与湍流特征提取及NFT标签生成

# Fractal and Turbulent Feature Extraction and NFT Label Generation for Pollock Style Migration Paintings Based on VGG19

本文提出了一种创新的方法，融合了深度学习、分形分析和湍流特征提取技术，以创作波洛克风格的抽象艺术品。图像的内容和风格特征由MindSpore深度学习框架和预训练的VGG19模型提取。然后采用优化过程。该方法通过结合内容损失、风格损失和全方差损失来生成高质量的波洛克风格图像，以实现精确的风格迁移。此外，本文实现了一种基于差分盒计数法的分形维数计算方法，通过边缘提取和分形分析有效地估计了图像的分形维数。该方法基于二维离散小波变换，使用Haar小波对图像进行分解，以提取不同的频率信息。随后，结合多种特征生成独特的非同质令牌（NFT）标签，用于数字艺术品的认证和保护。实验结果表明，生成的艺术品在分形维数和湍流特征方面表现出显著的多样性和复杂性，而生成的NFT标签确保了每个数字收藏的唯一性和可篡改性。本方法将计算机视觉、数字信号处理和区块链技术有机结合，为数字艺术品的创作和认证提供了一种新的解决方案。

关键词：神经风格迁移、分形分析、湍流特征、NFT标签、深度学习

#### 测试环境:  

```markdown
- MindSpore 2.2.14  
- Python 3.8 (ubuntu18.04)  
- Cuda 11.1  
- RTX 3080x2 (20GB) * 1  
- 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz  
```

权重文件: [https://download.pytorch.org/models/vgg19-dcbb9e9d.pth](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)  

图片数据集: [https://xihe.mindspore.cn/datasets/wyqmath/MindSpore_pollock/](https://xihe.mindspore.cn/datasets/wyqmath/MindSpore_pollock/)  

#### 准备工作：  

1. 安装MindSpore框架 (对于不同环境，参考昇思文档进行安装: [https://www.mindspore.cn/install](https://www.mindspore.cn/install) )  

```bash  
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 检查是否安装成功（不同环境的检测方式不一致，参考昇思文档进行检查: [https://www.mindspore.cn/install](https://www.mindspore.cn/install))  

```bash  
python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
```

3. 如果为以下输出即为安装成功

```bash  
MindSpore version: 2.2.14
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!
```

4. 转换权重文件

```bash
python pth2ms.py -i vgg19.pth -m vgg
```

#### 运行：

先修改风格图片、内容图片的路径，根据需要修改迭代次数与输出。运行transfer.py文件即可

```bash
python transfer.py
```

看到类似于以下的输出即为正常运行

```bash
python transfer.pyitera: 0, total loss: -1.2639e+14
con_loss: 0.0000e+00, sty_loss: 2.4509e-05, TV_loss: 3.1324e-02
tex_loss: 3.6853e-10, drip_loss: 1.0373e-10
var_img mean: 0.3823, std: 0.2441
time: 6.10 seconds

itera: 100, total loss: -1.3058e+14
con_loss: 1.2713e-09, sty_loss: 1.6112e-05, TV_loss: 2.9808e-02
tex_loss: 3.6062e-10, drip_loss: -3.5794e-12
var_img mean: 0.4546, std: 0.2624
time: 25.41 seconds

itera: 200, total loss: -1.4156e+14
con_loss: 5.5112e-09, sty_loss: 5.3572e-06, TV_loss: 3.1334e-02
tex_loss: 5.5161e-10, drip_loss: 5.8940e-12
var_img mean: 0.5266, std: 0.3061
time: 44.29 seconds
```

感谢MindSpore社区的支持！