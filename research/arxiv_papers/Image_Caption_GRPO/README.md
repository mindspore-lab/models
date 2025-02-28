# image_caption_grpo
## 项目介绍
本项目是Group Relative Policy Optimization for Image Captioning论文的实现, 论文链接为<https://arxiv.org/abs/2503.01333>

## 环境信息
本项目在GPU上完成训练并进行评测, 详细环境信息如下
Cuda 11.6  
RTX 3090  
python==3.8.10  
mindspore==2.2.14  
mindnlp==0.4.0

## 项目结构
log/: 记录训练与评测的日志文件  
model_save/: 保存文件权重  
module/: 包含解码方法, resnet网络定义, 以及一些自定义工具包  
PreTrainedModel/: 保存的词表以及预训练权重文件  

config.py: 配置文件, 可以根据需要进行修改  
eval.py: 测试文件  
model.py：图像描述模型  
read_file.py: 读取图片与文本, 进行数据处理文件  
train_with_grpo.py: 基于grpo强化学习算法的训练, 包含grpo强化学习算法的实现  
train_with_scst.py：基于scst强化学习算法的训练, 包含scst强化学习算法的实现  
train.py: 基于交叉熵的训练

## 项目使用
1. 从此[链接](https://pan.baidu.com/s/1NT3Og0NQBGL4Kfca7Rc52w)下载数据集, 评测文件, 以及java包, 将数据集与评测文件解压到该项目目录下, 密码为fs2x  
2. 从此[链接](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/application/resnet50_224_new.ckpt)下载resnet预训练权重并放入PreTrainedModel目录下  
3. 评测需要java环境, 从1中下载的java包进行解压获得java目录, 解压后将"You Java Path"替换为java目录所在路径, 本实验在Linux-ubuntu系统下进行, 因此可以将下面内容复制到/etc/profile文件中, 然后执行source /etc/profile命令完成java环境配置(其他系统请查阅相关资料, 可以通过java -version判断java环境是否配置成功)  
    #set java env  
    export JAVA_HOME="You Java Path"  
    export JRE_HOME=${JAVA_HOME}/jre  
    export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib  
    export PATH=${JAVA_HOME}/bin:$PATH  
4. 训练命令:   
    (a) 第一阶段交叉熵优化训练命令:  
        nohup python -u train.py > log/train.log 2>&1 &  
    (b) 将第一阶段保存的最后一个ckpt文件初始化为SCST与GRPO的初始化权重文件(修改config.py文件中的self.ck), 并分别进行SCST与GRPO强化学习训练, 训练命令为  
        nohup python -u train_with_scst.py > log/train_with_scst.log 2>&1 &  
        nohup python -u train_with_grpo.py > log/train_with_grpo.log 2>&1 &  
5. 测试命令: nohup python -u eval.py > log/eval.log 2>&1 &