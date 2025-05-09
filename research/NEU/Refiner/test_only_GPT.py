#!/usr/bin/env python
# coding: utf-8

import warnings
import random
import torch
import pandas as pd
from tqdm import tqdm
from docre.processData import return_rel2dict, return_templates, return_docred
from docre.processLogits import return_doc_logits_2024
from docre.evaluation import evaluate

random.seed(527)
warnings.filterwarnings("ignore")  # Ignore all warnings

# Define paths
dataset_path = "./dataset"
rel_templates_path = "./mydata"
logits_path = "./slm/2021-atlop/docred/dev"

# Load data and models
p_to_num, num_to_p, p_to_name, name_to_p = return_rel2dict(dataset_path + "/docred/rel_info.json")
print(f"P159 CLSNUM: {p_to_num['P159']}\nP159 NAME: {p_to_name['P159']}")

titles, entities, entity_types, entity_indexs, documents_raw, relations = return_docred(dataset_path + "/docred/dev.json")
p2templates = return_templates(rel_templates_path + "/rel_templates.xlsx")
print(p2templates['P159'])

dreeam_relations = return_doc_logits_2024(
    test_data_path=dataset_path + "/docred/dev.json",
    rel2id_path=dataset_path + "/meta/rel2id.json",
    logits_path=logits_path + "/dev_logits.json"
)

# Prepare inputs and prompts
inputs = []
completions = []
statements = []
TOP_K = 4

for i in range(len(documents_raw)):
    entity_pairs = {(a, b): ['Na'] for a in range(len(entities[i])) for b in range(len(entities[i])) if a != b}
    for relation in relations[i]:
        if 'Na' in entity_pairs[(relation['h'], relation['t'])]:
            entity_pairs[(relation['h'], relation['t'])] = [relation['r']]
        else:
            entity_pairs[(relation['h'], relation['t'])].append(relation['r'])

    questions, prompts, answers = [], [], []

    for pair in entity_pairs.keys():
        question, answer = [], []
        logits = dreeam_relations[i][pair]
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
                         f"{chr(ord('A') + len(question))}. None of the above options is correct."))
        questions.append(question)
        answers.append(answer if answer else f"{chr(ord('A') + len(question) - 1)}")

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

# Define get_completion function for GPT-3 predictions
def get_completion(prompt, model="gpt-3.5-turbo-0613", temperature=0.0, max_tokens=1):
    import openai
    openai.api_key = "your-api-key"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=527,
    )
    return response.choices[0].message["content"]

# Evaluate model with tqdm progress bar
final_titles, final_h_idxs, final_t_idxs, final_rs = [], [], [], []

for j, (prompts, answers, questions) in enumerate(tqdm(zip(inputs, completions, statements), total=len(inputs), desc="Testing docred-dev document with GPT-3...")):
    for prompt, answer, question in zip(prompts, answers, questions):
        try:
            predict = get_completion(prompt)[0]
        except Exception as e:
            print(f"Connection Error of document {j + 1}: {e}")
            continue

        if len(predict) == 1 and ord('A') <= ord(predict) < ord('A') + len(question):
            idx = ord(predict) - ord('A')
            final_titles.append(question[idx][0]['title'])
            final_h_idxs.append(question[idx][0]['h'])
            final_t_idxs.append(question[idx][0]['t'])
            final_rs.append(question[idx][0]['r'])

# Save results and evaluate
df_result = pd.DataFrame(zip(final_titles, final_h_idxs, final_t_idxs, final_rs), columns=['title', 'h_idx', 't_idx', 'r'])
df_result.to_json("./dev_result_GPT_atlop.json", orient='records')

evaluate(
    data_path=dataset_path + "/docred",
    test_data="dev.json",
    result_data="./dev_result_GPT_atlop.json",
    output_path="./"
)
