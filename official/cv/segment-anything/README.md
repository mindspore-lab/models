# Segment Anything

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

## Installation

The code requires `python>=3.7` and supports Ascend platform, some important pre-dependencies is: 
1. mindspore: Please follow the instructions [here](https://www.mindspore.cn/install) to install mindspore dependencies.
2. mindformers: please follow the instructions [here](https://gitee.com/mindspore/mindformers) using source code to install mindformers, 

Clone the repository locally and install with

```shell
git clone https://github.com/Mark-ZhouWX/models.git
cd models/official/cv/segment-anything
pip install -r requirements.txt
```

## Finetune

Finetune is a popular method that adapts large pretrained model to specific downstream tasks. Currently, finetune with box-prompt and text-prompt is supported.

*Note that finetune of SAM is not open-source at [official implementation of pytorch](https://github.com/facebookresearch/segment-anything). 
In this repository, finetune is an experimental function and still under improvement*

### Finetune with box-prompt
The bounding boxes are used as prompt input to predict mask.
Beside fine-tuning our code on COCO2017 dataset which contains common seen objects and lies in the similar distribution of the original [training dataset](https://segment-anything.com/dataset/index.html) of SAM, We have done further experiments on a medical imaging segmentation dataset [FLARE22](https://flare22.grand-challenge.org/Dataset/). Result shows that the finetune method in this repository is effective.

The bellowing shows the mask quality before and after finetune.


| pretrained_model | dataset  |    epochs     | mIOU |
|:----------------:| -------- |:-------------:|------|
|    sam-vit-b     | COCO2017 | 0 (zero-shot) | 77.4 |
|    sam-vit-b     | COCO2017 |      20       | 83.5 |
|    sam-vit-b     | FLARE22  | 0 (zero-shot) | 79.5 |
|    sam-vit-b     | FLARE22  |      10       | 88.1 |

A machine with **32G ascend memory** is required for box-prompt finetune.

for standalone finetune of COCO dataset, please run:
```shell
python train.py -c configs/coco_box_finetune.yaml -o amp_level=O2
```

for distributed finetune of COCO dataset, please run:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c configs/coco_box_finetune.yaml -o amp_level=O2
```
the fine-tuned model will be saved at the work_root specified in `configs/coco_box_finetune.yaml`. to eval the model, please run:
```shell
python eval.py -c configs/coco_box_finetune.yaml -o amp_level=O2 network.model.checkpoint=your/path/to/ckpt
```
for a fast single image inference, please run,
```shell
python inference.py --amp_level=O2 --checkpoint=your/path/to/ckpt
```

The original FLARE22 dataset contains image in 3D format and ground truth labelled as instance segmentation ids. Run

```shell
python scripts/preprocess_CT_MR_dataset.py
```

to preprocess it to the format of 2D RGB image and binary mask

The following steps are similar to COCO dataset finetune, please refer to the aforementioned description.

Here are the examples of segmentation result predicted by box-prompt fine-tuned SAM:

<div align="center">
<img src="images/coco_bear.jpg" height="350" />
    
<img src="images/flare_organ.jpg" height="350" />
</div>

<p align="center">
  <em> COCO2017 image example</em>
                        
                        
  <em> FLARE22 image example </em>
</p>


### Finetune with text-prompt
*Note again that text-to-mask finetune is exploratory and not robust, and the official pytorch code is not release yet.*


The training procedure described in the official SAM paper is quite interesting that does not require new text annotation. Specifically, for each manually collected mask with area larger than 1002 we extract the CLIP image embedding. Then, during training, we prompt SAM
with the extracted CLIP image embeddings as text prompt input. At inference time we run text through CLIP’s text encoder and then give the resulting text embedding as a prompt to SAM

The key that make the training procedure work is that CLIP’s image embeddings are trained to align with its text embeddings.

This repository provides an implementation of text-to-mask finetune referring to the model structure and training procedure described in the official SAM paper and replace CLIP to a stronger multimodal encoder BLIP2.

A machine with **64G ascend memory** is required for text-prompt finetune.

First download  SA-1B dataset and put it under `${project_root}/datasets/sa-1b`.

for standalone finetune of SA-1B dataset, please run:
```shell
python train.py -c configs/sa1b_text_finetune.yaml
```
the BLIP2 checkpoint and bert vocabulary.txt will be automatically downloaded at `./checkpoint_download/`

for distributed finetune, please run:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c configs/sa1b_text_finetune.yaml
```
the fine-tuned model will be saved at the work_root specified in `configs/sa1b_text_finetune.yaml`. For a fast single image inference, please run,

```shell
python text_inference.py --checkpoint=your/path/to/ckpt
```

## Demo

First download the weights ([sam_vit_b](https://download.mindspore.cn/toolkits/mindone/sam/sam_vit_b-35e4849c.ckpt), [sam_vit_l](https://download.mindspore.cn/toolkits/mindone/sam/sam_vit_l-1b460f38.ckpt), [sam_vit_h](https://download.mindspore.cn/toolkits/mindone/sam/sam_vit_h-c72f8ba1.ckpt)) and put them under `${project_root}/models` directory.
There are two recommended ways to use sam.

### Using sam with prompts

SAM predicts object masks given prompts that indicate the desired object. if a point prompt is given, three plausible masks are generated.

```shell
python demo/inference_with_promts.py --prompt-type point --model-type vit_h
```

<p float="left">
    <img src=images/truck_mask1.png width="400"/><img src=images/truck_mask2.png width="400"/><img src=images/truck_mask3.png width="400"/>
</p>

If a prompt with two points is given, one plausible mask is generated instead of 3 because of less ambiguity compared to one point prompt.
The star in green and red denotes positive and negtive point, respectively.

<div align="center">
    <img alt="img.png" src="images/truck_two_point.png" width="600"/>
</div>

If a box prompt is given, one plausible masks is generated.

```shell
python demo/inference_with_promts.py --prompt-type box --model-type vit_h
```

<div align="center">
    <img alt="img.png" width="600" src="images/truck_box.png"/>
</div>

If a prompt with both a box and a point is given, one plausible mask is generated.

```shell
python demo/inference_with_promts.py --prompt-type point_box --model-type vit_h
```

<div align="center">
    <img alt="img.png" width="600" src="images/truck_point_box.png"/>
</div>

See `python demo/inference_with_promts.py --help` to explore more custom settings.

### Using sam with Automatic Mask Generation(AMG)

Since SAM can efficiently process prompts, masks for the entire image can be generated by sampling a large number of prompts over an image. AMG works by sampling single-point input prompts in a grid over the image, from each of which SAM can predict multiple masks. Then, masks are filtered for quality and deduplicated using non-maximal suppression. Additional options allow for further improvement of mask quality and quantity, such as running prediction on multiple crops of the image or postprocessing masks to remove small disconnected regions and holes.

```shell
python demo/inference_with_amg.py --model-type vit_h
```

<div align="center">
<img src="images/dengta.jpg" height="350" />
      
<img src="images/dengta-amg-vith.png" height="350" />
</div>

See `python demo/inference_with_amg.py --help` to explore more custom settings.



