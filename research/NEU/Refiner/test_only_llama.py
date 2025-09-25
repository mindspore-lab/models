#!/usr/bin/env python
# coding: utf-8

import warnings
import random
from mindformers import build_context
from mindformers import AutoModel, AutoTokenizer, pipeline, TextStreamer
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
meta_llama_2_7b_path = c2net_context.pretrain_model_path + "/" + "Meta-Llama-2-7B"
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



# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained(meta_llama_2_7b_path)

# 模型实例化，需要提前将模型转成 ckpt 的，可以参考 https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html
# 修改成本地的权重路径
model = AutoModel.from_pretrained('llama2_7b', checkpoint_name_or_path= meta_llama_2_7b_path + "/llama2_7b.ckpt", use_past=True)


# Evaluate model
final_titles, final_h_idxs, final_t_idxs, final_rs = [], [], [], []

for prompts, answers, questions in tqdm(zip(inputs, completions, statements), total=len(inputs), desc="Testing..."):
    for prompt, answer, question in zip(prompts, answers, questions):
        # pipeline启动非流式推理任务
        text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
        output = text_generation_pipeline([prompt], do_sample=False, top_p=0.9)
        
        predicts = output[0][len(prompt):]
        predicts = predicts.replace(' ', '').split(',')

        for predict in predicts:
            if len(predict) == 1 and ord(predict) >= ord('A') and (ord(predict) - ord('A') + 1) < len(question):
                final_titles.append(question[ord(predict) - ord('A')][0]['title'])
                final_h_idxs.append(question[ord(predict) - ord('A')][0]['h'])
                final_t_idxs.append(question[ord(predict) - ord('A')][0]['t'])
                final_rs.append(question[ord(predict) - ord('A')][0]['r'])

df_result = pd.DataFrame(zip(final_titles, final_h_idxs, final_t_idxs, final_rs), columns=['title', 'h_idx', 't_idx', 'r'])
df_result.to_json("dev_result_llama2_atlop.json", orient='records')

from docre.evaluation import evaluate

evaluate(
    data_path=dataset_path + "/docred",
    test_data="dev.json",
    result_data="./dev_result_llama2_atlop.json",
    output_path="./"
)
