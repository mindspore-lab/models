# AGL-SC

<p align="center">
    <font size=6><strong>💬AGL-SC: Amplify Graph Learning for Recommendation via Sparsity Completion</strong></font>
</p>


The project is an implementation for AGL-SC model, which has firstly been proposed in the paper [💬Amplify Graph Learning for Recommendation via Sparsity Completion]. 

The implementations are based on [mindspore](https://gitee.com/mindspore). 


## Brief Introduction
AGL-SC is proposed to enhance the graph structure for CF more effectively, thereby optimizing the representation of graph nodes.

It utilize graph neural network and variational inference to mine multi-order features, integrated to implement the completion and enhancement of missing graph structures.

## Usage

**1. Project Preparation**:

Download this project and unzip.

**3. Running Preparation**:

Install all required packages.  
Adjust the parameters in ```ml.py``` python file. 

```
cd code
vim ml.py
# find and set the 'config' , save and quit
```
**Hint**:  
Pay attention to ```config["dataset"]``` parameter means the the folder naming of the dataset in ```./data``` folder. Please download the dataset and process it like LightGCN then place it in ```./data``` folder.


**4. Running Model**:
 
Run ```ml.py``` for *training*, *validation*, and *testing*.  

```
cd code
python ml.py
```
