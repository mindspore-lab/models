# DocREfiner

The code includes performance test scripts for evaluating only-LLM methods.

## `test_only_llama.py`

### Code Overview

This code uses Llama-2-7B to perform document-level relation extraction. The provided code demo is shown to process datasets, predict relations using the ATLOP SLM logits, and evaluate the results.

### Requirements

* Python 3.7+
* mindformers
* scikit-learn
* tqdm
* pandas
* docre (custom module)
* c2net (custom module)

### Installation

Install the required Python packages using pip:

```bash
pip install mindspore mindformers scikit-learn tqdm pandas openpyxl
```

Ensure the custom module `docre` is available in your Python path. Note that `c2net` is not necessary to use, you just need to replace all the paths with the `c2net` involved with your local paths.

### Usage

1. **Prepare the Environment**: Initialize the data context and set paths for datasets and pretrained models.
2. **Load Data and Models**: Use the provided functions to load datasets, relation templates, and pre-trained model logits.
3. **Generate Prompts**: Construct prompts and inputs for the model based on the loaded data.
4. **Run the Model**: Use the LLaMA2-7B model to generate predictions.
5. **Evaluate Results**: Save the model's predictions and evaluate them against the ground truth using the provided evaluation function.

#### Running the Script

Execute the script with Python:

```bash
python test_only_llama.py
```

The script will process the data, generate prompts, run the model, and evaluate the results. Output predictions will be saved to `dev_result_llama2_atlop.json`.

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
