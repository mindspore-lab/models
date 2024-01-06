# SPEECH 🚀

<p align="center">
    <font size=6><strong>💬SPEECH: Structured Prediction with Energy-Based Event-Centric Hyperspheres</strong></font>
</p>


🍎 The project is an official implementation for [**SPEECH**](https://github.com/zjunlp/SPEECH) model and a repository for [**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip) dataset, which has firstly been proposed in the paper [💬SPEECH: Structured Prediction with Energy-Based Event-Centric Hyperspheres](https://arxiv.org/abs/2305.13617) accepted by ACL 2023 main conference. 

🖥️ We also release the [poster](https://github.com/zjunlp/SPEECH/tree/main/ACL2023@Poster_Speech.pdf) and [slides](https://github.com/zjunlp/SPEECH/tree/main/ACL2023@Slides_Speech.pdf) for better understanding of this paper. 

🤗 The implementations are based on [mindspore](https://gitee.com/mindspore) and also referred to [OntoED](https://github.com/231sm/Reasoning_In_EE) & [DeepKE](https://github.com/zjunlp/DeepKE). 

🤗 The baseline implementations are reproduced with codes referred to [MAVEN's baselines](https://github.com/THU-KEG/MAVEN-dataset/) or with official implementation. 


## Brief Introduction 📣
SPEECH is proposed to address event-centric structured prediction with energy-based hyperspheres.  
SPEECH models complex dependency among event structured components with energy-based modeling, and represents event classes with simple but effective hyperspheres.


## Project Structure 🔍
The structure of data and code is as follows: 

```shell
SPEECH
├── README.md
├── requirements.txt    # for package requirements
├── src
│   ├── data_utils_ms.py       # for data processing
│   ├── speech_ms.py		        # main model (bert serves as the backbone)
│   ├── run_speech_ms.py	# for model running
├── sripts
│   ├── run_speech_ms.sh	# bash file for model running 
└── Datasets		    # data (git clone from github)
    ├── MAVEN_ERE   
    │   ├── train.jsonl     # for training
    │   ├── test.jsonl      # for testing
    │   └── valid.jsonl     # for validation
    ├── OntoEvent-Doc
    │   ├── event_dict_label_data.json      # containing all event type labels 
    │   ├── event_dict_on_doc_train.json	# for training
    │   ├── event_dict_on_doc_test.json		# for testing
    │   └── event_dict_on_doc_valid.json	# for validation
    └── README.md 
```

## Requirements 📦

- python==3.9.12

- mindspore==2.2.0

- mindnlp==0.2.0 

- scikit-learn==1.2.2

- sentencepiece==0.1.99


## Usage 🛠️

**1. Project Preparation**:

Download this project and unzip the [dataset](https://github.com/zjunlp/SPEECH/tree/main/Datasets). You can directly download the archive


**2. Data Preparation**: 

Unzip [**MAVEN_ERE**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/MAVEN_ERE.zip) and [**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip) datasets stored at ```./Datasets```. 
 
```
cd Datasets/
unzip MAVEN_ERE
unzip OntoEvent-Doc
cd .. 
```


**3. Running Preparation**:

Install all required packages.  
Adjust the parameters in [```run_speech.sh```](https://github.com/zjunlp/SPEECH/tree/main/run_speech.sh) bash file. 

```
pip install -r requirements.txt
vim run_speech.sh
# input the parameters, save and quit
```
**Hint**:  
- Please refer to ```main()``` function in [```run_speech.py```](https://github.com/zjunlp/SPEECH/tree/main/run_speech.py) file for detail meanings of each parameters.
- Pay attention to ```--ere_task_type``` parameter candidates:  
    - "doc_all" is for "All Joint" experiments in the paper 
    - "doc_joint" is for each ERE subtask "+joint" experiments in the paper
    - "doc_temporal"/"doc_causal/"doc_sub" is for each ERE subtask experiments only 
- Note that the loss ratio λ1, λ2, λ3, for trigger classification, event classification and event-relation extraction depends on different tasks, please ensure a correct setting of these ratios, referring to line 56-61 in [```speech_ms.py```](https://github.com/zjunlp/SPEECH/tree/main/speech.py) and [```speech_distilbert_ms.py```](https://github.com/zjunlp/SPEECH/tree/main/speech_distilbert.py) file for details. We also present the loss ratio setting in Appendix B in our paper.  



**4. Running Model**:
 
Run [```./run_speech_ms.sh```](https://github.com/zjunlp/SPEECH/tree/main/run_speech.sh) for *training*, *validation*, and *testing*.  

```
./run_speech.sh

# Or you can run run_speech.py with manual parameter input in the terminal.

python run_speech.py --para... 
```
**Hint**:  
- A folder of model checkpoints will be saved at the path you input (```--output_dir```) in the bash file [```run_speech_ms.sh```](https://github.com/zjunlp/SPEECH/tree/main/run_speech.sh) or the command line in the terminal. 
- We also release the [checkpoints](https://drive.google.com/drive/folders/18gFW_m02pgiGV2piktS308w41iBRZeN2?usp=sharing) for direct testing (Dismiss ```--do_train``` in the parameter input)


## How about the Dataset 🗃️
We briefly introduce the datasets in Section 4.1 and Appendix A in our paper. 

[**MAVEN_ERE**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/MAVEN_ERE.zip) is proposed in a [paper](https://aclanthology.org/2022.emnlp-main.60) and released in [GitHub](https://github.com/THU-KEG/MAVEN-ERE).

[**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip), formatted in document level, is derived from [OntoEvent](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) which is formatted in sentence level. 

### Statistics
The statistics of ***MAVEN-ERE*** and ***OntoEvent-Doc*** are shown below, and the detailed data schema can be referred to [```./Datasets/README.md```]. 

Dataset         | #Document | #Mention | #Temporal | #Causal | #Subevent |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
MAVEN-ERE        | 4,480 | 112,276 | 1,216,217 | 57,992  | 15,841 |
OntoEvent-Doc    | 4,115 | 60,546 | 5,914 | 14,155 | / |

### Data Format
The data schema of MAVEN-ERE can be referred to their [GitHub](https://github.com/THU-KEG/MAVEN-ERE). 
Experiments on MAVEN-ERE in our paper involve:  
- 6 temporal relations: BEFORE, OVERLAP, CONTAINS, SIMULTANEOUS, BEGINS-ON, ENDS-ON
- 2 causal relations: CAUSE, PRECONDITION 
- 1 subevent relation: subevent\_relations

Experiments on OntoEvent-Doc in our paper involve:  
- 3 temporal relations: BEFORE, AFTER, EQUAL 
- 2 causal relations: CAUSE, CAUSEDBY

We also add a NA relation to signify no relation between the event mention pair for the two datasets. 

🍒 The OntoEvent-Doc dataset is stored in json format. Each *document* (specialized with a *doc_id*, e.g., 95dd35ce7dd6d377c963447eef47c66c) in OntoEvent-Doc datasets contains a list of "events" and a dictionary of "relations", where the data format is as below:

```
[a doc_id]:
{
    "events": [
    {
        'doc_id': '...', 
        'doc_title': 'XXX', 
        'sent_id': , 
        'event_mention': '......', 
        'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
        'trigger': '...', 
        'trigger_pos': [, ], 
        'event_type': ''
    },
    {
        'doc_id': '...', 
        'doc_title': 'XXX', 
        'sent_id': , 
        'event_mention': '......', 
        'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
        'trigger': '...', 
        'trigger_pos': [, ], 
        'event_type': ''
    },
    ... 
    ],
    "relations": { // each event-relation contains a list of 'sent_id' pairs.  
        "COSUPER": [[,], [,], [,]], 
        "SUBSUPER": [], 
        "SUPERSUB": [], 
        "CAUSE": [[,], [,]], 
        "BEFORE": [[,], [,]], 
        "AFTER": [[,], [,]], 
        "CAUSEDBY": [[,], [,]], 
        "EQUAL": [[,], [,]]
    }
} 
```


## How to Cite 📝
📋 Thank you very much for your interest in our work. If you use or extend our work, please cite the following paper:

```bibtex
@inproceedings{ACL2023_SPEECH,
    author    = {Shumin Deng and
               Shengyu Mao and
               Ningyu Zhang and
               Bryan Hooi},
  title       = {SPEECH: Structured Prediction with Energy-Based Event-Centric Hyperspheres},
  booktitle   = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  publisher   = {Association for Computational Linguistics},
  pages       = {},
  year        = {2023},
  url         = {https://arxiv.org/abs/2305.13617}
}
```
