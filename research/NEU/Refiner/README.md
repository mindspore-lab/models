# Refiner


## Overview

Our code includes performance test scripts for evaluating only-LLM methods, as well as scripts for refiner using LLMs.
* `test_only_llama.py`
* `test_only_GPT.py`
* `test_refiner.py`

## `test_only_llama.py`

### Code Overview

This code uses `Llama-3-8B-Instruct` to perform document-level relation extraction. The provided code demo is shown to process datasets, predict relations using the ATLOP SLM logits, and evaluate the results.

### Requirements

* Python 3.7+
* PyTorch
* Transformers
* scikit-learn
* tqdm
* pandas
* docre (custom module)
* c2net (custom module)

### Installation

Install the required Python packages using pip:

```bash
pip install torch transformers scikit-learn tqdm pandas openpyxl
```

Ensure the custom module `docre` is available in your Python path. Note that `c2net` is not necessary to use, you just need to replace all the paths with the `c2net` involved with your local paths.

### Usage

1. **Prepare the Environment**: Initialize the data context and set paths for datasets and pretrained models.
2. **Load Data and Models**: Use the provided functions to load datasets, relation templates, and pre-trained model logits.
3. **Generate Prompts**: Construct prompts and inputs for the model based on the loaded data.
4. **Run the Model**: Use the LLaMA3-8B model to generate predictions.
5. **Evaluate Results**: Save the model's predictions and evaluate them against the ground truth using the provided evaluation function.

#### Running the Script

Execute the script with Python:

```bash
python test_only_llama.py
```

The script will process the data, generate prompts, run the model, and evaluate the results. Output predictions will be saved to `dev_result_llama3_instruct_atlop.json`.

#### Example Output

The script prints example inputs and completions, showing the format of the processed data and the model's predictions.

```plaintext
INSTRUCTION: Read the DOCUMENT and answer the QUESTION. Write the answers in ANSWER.
DOCUMENT: ...
QUESTION: Which of the following is right?
...
ANSWER: 
```

#### Evaluation

After running the script, the results are evaluated using the `evaluate` function, which compares the model's predictions with the ground truth and outputs performance metrics.

### Notes

- Ensure the `dataset_path`, `pretrain_model_path`, and other paths are correctly set according to your environment.
- Modify the top-k variable to change the number of top predictions considered.
- The script is set to ignore warnings for cleaner output.

## `test_only_GPT.py`

### Overview

This project utilizes `GPT-3` for document-level relation extraction. The provided code processes datasets, predicts relations using the ATLOP model, and evaluates the results.

### Requirements

- Python 3.7+
- PyTorch
- Transformers
- pandas
- tqdm
- OpenAI API key
- docre (custom module)
- c2net (custom module)

### Installation

Install the required Python packages using pip:

```bash
pip install torch transformers pandas tqdm openai
```

Ensure the custom module `docre` is available in your Python path. Note that `c2net` is not necessary to use, you just need to replace all the paths with the `c2net` involved with your local paths.

### Setup

1. **Set up OpenAI API Key**: Set your OpenAI API key in the `get_completion` function.
2. **Prepare the Environment**: Ensure the dataset and required files are available in the specified paths.

### Usage

#### Running the Script

Execute the script with Python:

```bash
python test_only_GPT.py
```

#### Parameters

- `dataset_path`: Path to the dataset directory.
- `rel_templates_path`: Path to the relationship templates directory.
- `logits_path`: Path to the logits directory.

#### Script Workflow

1. **Load Data**: Loads relation information, document data, and relation templates.
2. **Generate Prompts**: Constructs prompts and inputs for the model based on the loaded data.
3. **Run the Model**: Uses GPT-3 to generate predictions.
4. **Evaluate Results**: Saves the model's predictions and evaluates them against the ground truth using the provided evaluation function.

#### Example Output

The script prints example inputs and completions, showing the format of the processed data and the model's predictions.

```plaintext
##INSTRUCTION: Read the ##DOCUMENT and answer the ##QUESTION. Write the answers in ##ANSWER.
##DOCUMENT: ...
##QUESTION: Which of the following is right?
...
##ANSWER:
```

#### Evaluation

After running the script, the results are evaluated using the `evaluate` function, which compares the model's predictions with the ground truth and outputs performance metrics.

### Notes

- Ensure the `dataset_path`, `rel_templates_path`, and `logits_path` are correctly set according to your environment.
- Modify the `TOP_K` variable to change the number of top predictions considered.
- The script is set to ignore warnings for cleaner output.


## `test_refiner.py`

This project refines document-level relation extraction using `llama3-8b-instruct`. The script processes documents, refines relation extraction results, and evaluates the performance using a SLM original performance and refinement after LLM.

### Requirements

- Python 3.7+
- PyTorch
- Transformers
- TQDM
- Pandas

### Installation

1. Install the required Python packages:
    ```bash
    pip install torch transformers tqdm pandas
    ```

2. Clone this repository and navigate to the project directory.

### Usage

#### Prepare Data

1. Ensure your dataset is prepared and paths are correctly set in the script:
    - Dataset Path: `c2net_context.dataset_path + "/dataset"`
    - Relation Templates Path: `c2net_context.dataset_path + "/rel_templates"`
    - DocRED Logits Path: `c2net_context.dataset_path + "/docred-logits"`

2. Set the pre-trained model path:
    - Meta-Llama-3-8B-Instruct Path: `c2net_context.pretrain_model_path + "/Meta-Llama-3-8B-Instruct"`

3. Set the output path:
    - `c2net_context.output_path`
  
4. Ensure the custom module `docre` is available in your Python path. Note that `c2net` is not necessary to use, you just need to replace all the paths with the `c2net` involved with your local paths.

#### Run the Script

Execute the script to process and refine the documents:
```bash
python test_refiner.py
```

#### Example Output

The script prints example inputs and completions, showing the format of the processed data and the model's predictions.

```plaintext
##INSTRUCTION: Read the ##DOCUMENT and answer the ##QUESTION. Write the answers in ##ANSWER.
##DOCUMENT: ...
##QUESTION: Which of the following is right?
...
##ANSWER:
```

#### Evaluation

After running the script, the results are evaluated using the `evaluate` function, which compares the model's predictions with the ground truth and outputs performance metrics.

### Notes

- Ensure the `dataset_path`, `rel_templates_path`, and `logits_path` are correctly set according to your environment.
- Modify the `TOP_K` variable to change the number of top predictions considered.
- The script is set to ignore warnings for cleaner output.
  
