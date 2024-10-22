### VOC格式自定义数据集finetune流程

#### 数据集格式转换

voc格式的数据集，其文件目录通常如下所示：
```
             ROOT_DIR
                ├── annotations
                │        ├── 000000.xml
                │        └── 000002.xml
                ├── train.txt
                └── vallid.txt
                └── images
                        ├── 000000.jpg
                        └── 000002.jpg
```
annotations文件夹下的xml文件为每张图片的标注信息，主要内容如下：
```
<annotation>
  <folder>JPEGImages</folder>
  <filename>000377.jpg</filename>
  <path>F:\baidu\VOC2028\JPEGImages\000377.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>750</width>
    <height>558</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>hat</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>142</xmin>
      <ymin>388</ymin>
      <xmax>177</xmax>
      <ymax>426</ymax>
    </bndbox>
  </object>
```
其中包含多个object, object中的name为类别名称，xmin, ymin, xmax, ymax则为检测框左上角和右下角的坐标。

MindYOLO支持的数据集格式为YOLO格式。由于MindYOLO在验证阶段选用图片名称作为image_id，因此图片名称只能为数值类型，而不能为字符串类型，还需要对图片进行改名。数据集格式转换步骤如下：
* 将图片复制到相应的路径下并改名
* 在根目录下相应的txt文件中写入该图片的相对路径
* 解析xml文件，在相应路径下生成对应的txt标注文件
* 验证集生成最终的json文件

详细实现可参考[voc2yolo.py](./utils/voc2yolo.py)，运行方式如下：

  ```shell
  python utils/voc2yolo.py --root_dir /path_to_dataset
  ```
运行以上命令将在不改变原数据集的前提下，在同级目录生成名为YOLODataSet的yolo格式数据集，运行之前需要将代码中的category_set修改为实际的类别名称列表。
#### 权重转换
可通过utils/pdparams2ckpt.py将paddle上训练好的pdparams文件转换为ckpt文件，运行方式如下：
  ```shell
  python utils/pdparams2ckpt.py --pdparams_path /path_to_pdparams_file --ckpt_path /path_to_ckpt_file
  ```
#### 修改数据集配置信息
需要在configs/coco.yaml中根据数据集的实际信息做相应修改，主要包括dataset_name-数据集名称，train_set-保存训练集数据路径的txt文件，val_set-保存训练集数据路径的txt文件，
nc-数据集类别数，names-数据集类别名称列表

#### 启动训练与验证

* 在多卡NPU/GPU上进行分布式模型训练，以8卡为例，可通过--weight参数加载预训练权重:
  ```shell
  mpirun --allow-run-as-root -n 8 python train.py --config ./configs/hyp.yaml --is_parallel True --weight /path_to_ckpt/WEIGHT.ckpt
  ```

* 在单卡NPU/GPU/CPU上训练模型：

  ```shell
  python train.py --config ./configs/hyp.yaml --weight /path_to_ckpt/WEIGHT.ckpt
  ```

* 在单卡NPU/GPU/CPU上评估模型的精度：

  ```shell
  python test.py --config ./configs/hyp.yaml --weight /path_to_ckpt/WEIGHT.ckpt
  ```