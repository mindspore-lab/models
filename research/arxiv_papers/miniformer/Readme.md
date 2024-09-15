# Readme

code中为gpu上可以直接运行的代码，环境如下：

```
python==3.9.1
mindspore=2.2.14
mindnlp=0.3.1
```

NPU_code为可以在npu上运行的代码，环境一样，其余的训练测试等指令也一样。

- 训练：

```
python train_seq2seqsum.py --data_path 训练集 --name 名称 --epoch 训练轮数
```

- 测试

进入代码，修改读入的模型和对应的测试集，以及最终的输出文件

```
python decode.py 
```

- 评估：

进入代码，修改模型的输出文件夹

```
python eval.py
```

- 对比：

对比的Transformer可见：https://github.com/Veteranback/transformer
