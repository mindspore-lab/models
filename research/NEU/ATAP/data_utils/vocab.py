import json
from os.path import join

shared_vocab , lama_vocab = None,None
def init_vocab(args):
    global shared_vocab
    global lama_vocab
    shared_vocab = json.load(open(join(args.data_dir, '29k-vocab.json'),encoding='utf-8'))
    lama_vocab = json.load(open(join(args.data_dir, '34k-vocab.json'),encoding='utf-8'))
    # print(shared_vocab['roberta-large'])

def token_wrapper(args, token):
    if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
        return 'Ġ' + token
    else:
        return token

def get_vocab(model_name, strategy):
    if strategy == 'shared':
        if 'gpt' in model_name:
            # print(shared_vocab['gpt2-xl'])
            return shared_vocab['gpt2-add-tokens']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return shared_vocab['gpt2-add-tokens']
        #开始
        elif model_name == 'bert-base-cased' or model_name == 'bert-large-cased' or model_name == 'bert-base-uncased':
            return shared_vocab['add_tokens']
        #结束
        else:
            assert model_name in shared_vocab
            # print(shared_vocab[model_name])
            return shared_vocab[model_name]
    elif strategy == 'lama':
        if 'gpt' in model_name:
            return lama_vocab['gpt2-xl']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return lama_vocab['roberta-large']
        else:
            assert model_name in lama_vocab
            return lama_vocab[model_name]

def get_vocab_by_strategy(args, tokenizer):
    if args.vocab_strategy == 'original':
        return tokenizer.get_vocab()
    else:
        return get_vocab(args.model_name, args.vocab_strategy)


