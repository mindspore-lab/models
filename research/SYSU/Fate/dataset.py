import datasets
import random
import os

def get_line_from_dataset(dataset, line):
    # there are some topics and None in dataset
    if len(dataset['train'][line]['text']) < 128:
        line += 1
        return get_line_from_dataset(dataset, line)
    return dataset['train'][line]['text']

def get_inputs():
    if not os.path.exists('datasets/wikitext-103-v1'):
        os.makedirs('datasets/wikitext-103-v1')
        # load from HF
        dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')
        dataset.save_to_disk('datasets/wikitext-103-v1')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/wikitext-103-v1')
    
    # get_line = []
    # for _ in range(num_prompts):
    #     get_line.append(random.randint(0, 1801350-1))

    # inputs = ()
    # for line in get_line:
    #     input_ids = tokenizer(get_line_from_dataset(dataset, line), return_tensors="pt").input_ids
    #     inputs = inputs + (input_ids,) 
    # return inputs

# sum
def get_xsum_inputs():
    if not os.path.exists('datasets/xsum'):
        os.makedirs('datasets/xsum')
        # load from HF
        dataset = datasets.load_dataset('xsum')
        dataset.save_to_disk('datasets/xsum')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/xsum')

    return dataset['validation']

# sum
def get_samsum_inputs():
    if not os.path.exists('datasets/samsum'):
        os.makedirs('datasets/samsum')
        # load from HF
        dataset = datasets.load_dataset('Samsung/samsum')
        dataset.save_to_disk('datasets/samsum')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/samsum')

    return dataset['test']

# translate
def get_wmt_inputs():
    if not os.path.exists('datasets/wmt16'):
        os.makedirs('datasets/wmt16')
        # load from HF
        dataset = datasets.load_dataset("wmt16", "ro-en")
        dataset.save_to_disk('datasets/wmt16')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/wmt16')

    return dataset['test']

# mmlu
def get_mmlu_inputs():
    if not os.path.exists('datasets/mmlu'):
        os.makedirs('datasets/mmlu')
        # load from HF
        dataset = datasets.load_dataset("lighteval/mmlu", "all")
        dataset.save_to_disk('datasets/mmlu')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/mmlu')

    return dataset['test']

# gsm8k
def get_gsm8k_inputs():
    if not os.path.exists('datasets/gsm8k'):
        os.makedirs('datasets/gsm8k', exist_ok=True)
        # load from HF
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = datasets.load_dataset("gsm8k", "main", download_config=config)
        dataset.save_to_disk('datasets/gsm8k')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/gsm8k')

    return dataset['test']

# ChatGPT-prompts
def get_ChatGPT_prompts_inputs():
    if not os.path.exists('datasets/ChatGPT-prompts'):
        os.makedirs('datasets/ChatGPT-prompts')
        # load from HF
        dataset = datasets.load_dataset("MohamedRashad/ChatGPT-prompts")
        dataset.save_to_disk('datasets/ChatGPT-prompts')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/ChatGPT-prompts')

    return dataset['train']

# openai_humaneval
def get_openai_humaneval_inputs():
    if not os.path.exists('datasets/openai_humaneval'):
        os.makedirs('datasets/openai_humaneval')
        # load from HF
        dataset = datasets.load_dataset("openai/openai_humaneval")
        dataset.save_to_disk('datasets/openai_humaneval')
    else:
        # load from local
        dataset = datasets.load_from_disk('datasets/openai_humaneval')

    return dataset['test']

if __name__ == "__main__":
   dataset = get_openai_humaneval_inputs()
#    dataset = dataset['validation']
   for i in range(len(dataset)):
       if i < 5:
        # print(dataset[i])
        print(dataset[i])

   