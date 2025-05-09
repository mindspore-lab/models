#!/usr/bin/env python
# coding: utf-8

import warnings
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from c2net.context import prepare
from docre.processData import return_rel2dict, return_templates, return_docred
from docre.processLogits import return_doc_logits_2024

random.seed(527)
warnings.filterwarnings("ignore")

# Initialize data context
c2net_context = prepare()

# Define paths
dataset_path = c2net_context.dataset_path + "/" + "dataset"
rel_templates_path = c2net_context.dataset_path + "/" + "rel_templates"
docred_logits_path = c2net_context.dataset_path + "/" + "docred-logits"
meta_llama_3_8b_instruct_path = c2net_context.pretrain_model_path + "/" + "Meta-Llama-3-8B-Instruct"
output_path = c2net_context.output_path

# Load relation and template data
p_to_num, num_to_p, p_to_name, name_to_p = return_rel2dict(dataset_path + "/docred/rel_info.json")
titles, entities, entity_types, entity_indexs, documents_raw, relations = return_docred(dataset_path + "/docred/dev.json")
p2templates = return_templates(rel_templates_path + "/rel_templates.xlsx")
atlop_relations = return_doc_logits_2024(
    test_data_path=dataset_path + "/docred/dev.json",
    rel2id_path=dataset_path + "/meta/rel2id.json",
    logits_path=docred_logits_path + "/atlop/dev/dev_logits.json"
)

inputs, completions, statements = [], [], []
TOP_K = 4

for i in range(len(documents_raw)):
    entity_pairs = {(a, b): ['Na'] for a in range(len(entities[i])) for b in range(len(entities[i])) if a != b}
    for relation in relations[i]:
        entity_pairs[(relation['h'], relation['t'])] = [relation['r']]

    questions, prompts, answers = [], [], []
    for pair in entity_pairs.keys():
        question, answer = [], []
        logits = atlop_relations[i][pair]
        if list(logits.keys())[0] == 'Na':
            continue

        for j, logit in enumerate(list(logits.keys())[:TOP_K]):
            if logit in entity_pairs[pair] and logit != 'Na':
                answer.append(f"{chr(ord('A') + j)}")
            if logit == 'Na':
                continue
            head_name = entities[i][pair[0]][random.randint(0, len(entities[i][pair[0]]) - 1)]
            tail_name = entities[i][pair[1]][random.randint(0, len(entities[i][pair[1]]) - 1)]
            now_question = f"{chr(ord('A') + j)}. " + p2templates[logit]
            now_question = now_question.replace('<head>', f"{head_name}({entity_types[i][pair[0]]})")
            now_question = now_question.replace('<tail>', f"{tail_name}({entity_types[i][pair[1]]})")
            question.append(({'title': titles[i], 'h': pair[0], 't': pair[1], 'r': logit}, now_question))
        
        question.append(({'title': titles[i], 'h': pair[0], 't': pair[1], 'r': 'Na'},
                         f"{chr(ord('A') + j)}. None of the above options is correct."))
        questions.append(question)
        answers.append(answer if answer else f"{chr(ord('A') + j)}")

    for k, q in enumerate(questions):
        prompt = f"""##INSTRUCTION: Read the ##DOCUMENT and answer the ##QUESTION. Write the answers in ##ANSWER.
        
##DOCUMENT: {" ".join(documents_raw[i])}

##QUESTION: Which of the following is right?
{" ".join([q[1] for q in questions[k]])}
##ANSWER: """
        prompts.append(prompt)
     
    inputs.append(prompts)
    completions.append(answers)
    statements.append(questions)

print(inputs[0][1] + ", ".join(completions[0][1]))
print(statements[0][0])

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    meta_llama_3_8b_instruct_path,
    use_cache=False,
    use_flash_attention_2=False,
    device_map="auto"
)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(meta_llama_3_8b_instruct_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id

# Evaluate model
final_titles, final_h_idxs, final_t_idxs, final_rs = [], [], [], []

for prompts, answers, questions in tqdm(zip(inputs, completions, statements), total=len(inputs), desc="Testing..."):
    for prompt, answer, question in zip(prompts, answers, questions):
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        with torch.inference_mode():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=1, top_p=0.9, temperature=0)
        predicts = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
        predicts = predicts.replace(' ', '').split(',')

        for predict in predicts:
            if len(predict) == 1 and ord(predict) >= ord('A') and (ord(predict) - ord('A') + 1) < len(question):
                final_titles.append(question[ord(predict) - ord('A')][0]['title'])
                final_h_idxs.append(question[ord(predict) - ord('A')][0]['h'])
                final_t_idxs.append(question[ord(predict) - ord('A')][0]['t'])
                final_rs.append(question[ord(predict) - ord('A')][0]['r'])

df_result = pd.DataFrame(zip(final_titles, final_h_idxs, final_t_idxs, final_rs), columns=['title', 'h_idx', 't_idx', 'r'])
df_result.to_json("dev_result_llama3_instruct_atlop.json", orient='records')

from docre.evaluation import evaluate

evaluate(
    data_path=dataset_path + "/docred",
    test_data="dev.json",
    result_data="./dev_result_llama3_instruct_atlop.json",
    output_path="./"
)