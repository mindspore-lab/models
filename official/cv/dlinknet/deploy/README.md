##  D-Linknet推理

以下为模型在ascend 310推理的步骤

### 1 Requirements
   |mindspore|ascend driver|firmware|cann toolkit/kernel|mindspore lite|
   | :------: | :------: | :------: | :------: | :------: |
   |2.3.1|24.1.rc2.b090|7.3.T10.0.b090|7.3.T10.0.b528|2.3.1|
   ```shell
   pip install -r requirement.txt
   ```

### 2 安装MindSpore Lite
   MindSpore Lite官方页面请查阅：[MindSpore Lite](https://mindspore.cn/lite) <br>
   - 下载tar.gz包并解压，同时配置环境变量LITE_HOME,LD_LIBRARY_PATH,PATH
     ```shell
     tar -zxvf mindspore_lite-[xxx].tar.gz
     export LITE_HOME=/[path_to_mindspore_lite_xxx]
     export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
     export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
     export Convert=$LITE_HOME/tools/converter/converter/converter_lite
     ```
     LITE_HOME为tar.gz解压出的文件夹路径，请设置绝对路径
   - 安装whl包
     ```shell
     pip install mindspore_lite-[xxx].whl
     ```
### 3 模型转换 ckpt -> mindir
   - 训练完成的模型ckpt权重转为mindspore mindir文件，例如
     ```shell
     # 导出mindspore mindir模型
     python export.py --config=./configs/dlinknet34_config.yaml --trained_ckpt=./dlinknet34.ckpt --file_name=dlinknet34.mindir --file_format=MINDIR --batch_size=1
     ```
     参数说明:
     - trained_ckpt: 为训练好的ckpt权重文件
     - file_name: 为转换后的mindspore mindir文件
   - 为了加快推理时加载模型的速度，可以再把mindspore mindir文件转换成mslite mindir文件，例如
     ```shell
     # 把mindspore mindir模型转换成mslite mindir模型
     $Convert --fmk=MINDIR --modelFile=./dlinknet34.mindir --outputFile=./dlinknet34_lite  --saveType=MINDIR --optimize=ascend_oriented 
     ```
     参数说明:
     - modelFile: 为export生成的mindspore mindir文件
     - outputFile: 为转换生成的mslite mindir文件，默认会加扩展名mindir
### 4 推理评估

   ```shell
   python deploy/mslite_predict.py --mindir_path=./dlinknet34_lite.mindir --image_path=~/DeepGlobe_Road_Extraction_Dataset/valid --save_result=True -result_folder=./predict_result/  --image_label_path=~/DeepGlobe_Road_Extraction_Dataset/valid_mask/
   ```
   参数说明:
   - mindir_path: 为导出的mindspore mindir文件或者转换后的mslite mindir文件
   - image_path: 为验证集的路径
   - result_folder: 推理生成的文件保存路径
   - image_label_path: 为验证集label的路径

## mindir支持列表

| model name | cards | batch size | graph compile | step | img/s | img size | iou | ckpt | mindspore mindir | mindsport lite mindir|
|:----------:|:-----:|:----------:|:-------------:|:----:|:-----:|:--------:|:---:|:----:|:----------------:|:--------------------:|
|dlinknet34|1|1|0.94s|622|8.56|1024x1024|98%|[chpt]()|[mindir]()|[mindir]()|
