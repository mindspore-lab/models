from c2net.context import prepare
import warnings
from docre.processData import return_rel2dict, return_templates, return_docred
from docre.processLogits import return_doc_logits_2024
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# Initialize context and suppress warnings
c2net_context = prepare()
random.seed(527)
warnings.filterwarnings("ignore")

# Set dataset paths
dataset_path = c2net_context.dataset_path + "/dataset"
rel_templates_path = c2net_context.dataset_path + "/rel_templates"
docred_logits_path = c2net_context.dataset_path + "/docred-logits"

# Set pre-trained model paths
meta_llama_3_8b_instruct_path = c2net_context.pretrain_model_path + "/Meta-Llama-3-8B-Instruct"

# Set output path
output_path = c2net_context.output_path

# Load relation and entity data
p_to_num, num_to_p, p_to_name, name_to_p = return_rel2dict(dataset_path + "/docred/rel_info.json")
titles, entities, entity_types, entity_indexs, documents_raw, relations = return_docred(dataset_path + "/docred/dev.json")
p2templates = return_templates(rel_templates_path + "/rel_templates.xlsx")
atlop_relations = return_doc_logits_2024(
    test_data_path=dataset_path + "/docred/dev.json",
    rel2id_path=dataset_path + "/meta/rel2id.json",
    logits_path=docred_logits_path + "/atlop/dev/dev_logits.json"
)

# Prepare prompts and answers
inputs = []
completions = []
statements = []
TOP_K = 4

for i in range(len(documents_raw)):
    entity_pairs = {(a, b): ['Na'] for a in range(len(entities[i])) for b in range(len(entities[i])) if a != b}

    for relation in relations[i]:
        entity_pairs[(relation['h'], relation['t'])].append(relation['r'])

    questions = []
    prompts = []
    answers = []

    for pair in entity_pairs.keys():
        question = []
        answer = []
        logits = atlop_relations[i][pair]
        keys = list(logits.keys())[:TOP_K]
        logits_values = torch.tensor([logits[key] for key in keys])
        softmax_values = F.softmax(logits_values, dim=0)
        softmax_logits = {keys[j]: softmax_values[j].item() for j in range(len(keys))}

        if list(logits.keys())[0] == 'Na' and softmax_values[1] >= softmax_values[0] * 0.3:
            j = 0
            for logit in list(logits.keys())[:TOP_K]:
                if logit in entity_pairs[pair] and logit != 'Na':
                    answer.append(f"{chr(ord('A') + j)}")
                if logit == 'Na':
                    continue

                head_name = random.choice(entities[i][pair[0]])
                tail_name = random.choice(entities[i][pair[1]])
                head_type = entity_types[i][pair[0]]
                tail_type = entity_types[i][pair[1]]
                now_question = f"{chr(ord('A') + j)}. " + p2templates[logit].replace('<head>', f"{head_name}({head_type})").replace('<tail>', f"{tail_name}({tail_type})")
                question.append({'title': titles[i], 'h': pair[0], 't': pair[1], 'r': logit, 'score': softmax_logits[logit]}, now_question)
                j += 1

            question.append({'title': titles[i], 'h': pair[0], 't': pair[1], 'r': 'Na', 'score': softmax_logits['Na']}, f"{chr(ord('A') + j)}. None of the above options is correct.")
            questions.append(question)
            answers.append(answer if answer else [f"{chr(ord('A') + j)}"])

    for k in range(len(questions)):
        prompt = f"##INSTRUCTION: Read the ##DOCUMENT and answer the ##QUESTION. Write the answers in ##ANSWER.\n\n##DOCUMENT: {' '.join(documents_raw[i])}\n\n##QUESTION: Which of the following is right?\n" + '\n'.join(q[1] for q in questions[k]) + "\n##ANSWER: "
        prompts.append(prompt)

    inputs.append(prompts)
    completions.append(answers)
    statements.append(questions)

# Load model and tokenizer
model_id = meta_llama_3_8b_instruct_path
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id

# Evaluate and save results
slm_res = pd.read_json('./refine_atlop_dev/dev_results.json')
final_titles = slm_res['title'].tolist()
final_h_idxs = slm_res['h_idx'].tolist()
final_t_idxs = slm_res['t_idx'].tolist()
final_rs = slm_res['r'].tolist()

correct = 0
mis_correct = 0

for prompts, answers, questions in tqdm(zip(inputs, completions, statements), total=len(inputs), desc="Refine docred-dev document with ATLOP in llama3-8B-instruct..."):
    for prompt, answer, question in zip(prompts, answers, questions):
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=True, top_p=0.9, temperature=1.8, output_scores=True, return_dict_in_generate=True)
        generated_token_id = output.sequences[:, -1].item()
        scores = output.scores[0]
        probabilities = F.softmax(scores, dim=-1)
        all_probabilities = probabilities[0].tolist()

        for i, prob in enumerate(all_probabilities):
            if prob > 0:
                predict = tokenizer.decode([i]).split()
                if predict and predict[0] in ['A', 'B', 'C', 'D']:
                    question[ord(predict[0]) - ord('A')][0]['score'] += prob

        all_relations = [choice[0] for choice in question]
        threshold = all_relations[-1]['score']
        for i, now_relation in enumerate(all_relations):
            if now_relation['score'] > threshold and now_relation['r'] != 'Na':
                final_titles.append(now_relation['title'])
                final_h_idxs.append(now_relation['h'])
                final_t_idxs.append(now_relation['t'])
                final_rs.append(now_relation['r'])
                if chr(ord('A') + i) in answer:
                    correct += 1
                else:
                    mis_correct += 1

print(f"correct: {correct}  mis_correct: {mis_correct}")

df_result = pd.DataFrame(zip(final_titles, final_h_idxs, final_t_idxs, final_rs), columns=['title', 'h_idx', 't_idx', 'r'])
df_result.to_json("./data_analyze/refine_results.json", orient='records')

# Baseline evaluation
from docre.evaluation import evaluate

evaluate(
    data_path=dataset_path + "/docred",
    test_data="dev.json",
    result_data="./refine_atlop_dev/dev_results.json",
    output_path="./refine_atlop_dev"
)

# Refine result evaluation
evaluate(
    data_path=dataset_path + "/docred",
    test_data="dev.json",
    result_data="./data_analyze/refine_results.json",
    output_path="./data_analyze"
)
