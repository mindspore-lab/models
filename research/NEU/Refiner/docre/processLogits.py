import json
import pandas as pd
import numpy as np


def return_doc_logits(test_data_path = "dataset/docred/dev.json", 
                  rel2id_path = "../ass3/ATLOP/meta/rel2id.json", 
                  logits_path = "./dataNEW/res_slm_llm_dreeam/logits.json" 
                 ):
    
    test_data = open(test_data_path, 'r', encoding='utf-8')
    json_info = test_data.read()
    df = pd.read_json(json_info)

    rel2id = json.load(open(rel2id_path, 'r'))
    
    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        
    df_rel = pd.read_json(logits_path)
    
    
    doc_index = 0
    j = 0
    doc_relations = []
    doc_relation = {}
    while True:
        if j == len(df_rel['doc_index']):
            doc_relations.append(doc_relation)
            doc_relation = {}
            break

        if df_rel['doc_index'][j] != doc_index + 1:
            doc_index += 1
            doc_relations.append(doc_relation)
            doc_relation = {}
            continue

        sorted_numbers = sorted(enumerate(df_rel['relations'][j]), key=lambda x: x[1], reverse=True)

        sorted_names = [id2rel[index] for index, _ in sorted_numbers]
        sorted_values = [value for _, value in sorted_numbers]

        relations = {}
        for i in range(len(sorted_names)):
            relations[sorted_names[i]] = sorted_values[i]

        doc_relation[(df_rel['h_index'][j], df_rel['t_index'][j])] = relations

        j += 1
        
    return doc_relations


def return_doc_logits_2024(test_data_path = "dataset/docred/dev.json", 
                  rel2id_path = "../ass3/ATLOP/meta/rel2id.json", 
                  logits_path = "./dataNEW/res_slm_llm_dreeam/logits.json" 
                 ):
    
    
    test_data = open(test_data_path, 'r', encoding='utf-8')
    json_info = test_data.read()
    df = pd.read_json(json_info)
    

    rel2id = json.load(open(rel2id_path, 'r'))
    
    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        
    df_rel = pd.read_json(logits_path)
    
    
    doc_index = 0
    j = 0
    doc_relations = []
    doc_relation = {}
    while True:
        if j == len(df_rel['doc_idxs']):
            doc_relations.append(doc_relation)
            doc_relation = {}
            break

        if df_rel['doc_idxs'][j] != doc_index:
            doc_index += 1
            doc_relations.append(doc_relation)
            doc_relation = {}
            continue

        doc_relation[(df_rel['h_idx'][j], df_rel['t_idx'][j])] = dict(sorted(df_rel['logprobs_r'][j].items(), key=lambda item: item[1], reverse=True))

        j += 1
        
    return doc_relations



def process_data(data):
    sum_dict = {}
    count_dict = {}

    for sublist in data:
        for tup in sublist:
            second_num = tup[1]
            first_num = tup[0]

            if second_num in sum_dict:
                sum_dict[second_num] += first_num
                count_dict[second_num] += 1
            else:
                sum_dict[second_num] = first_num
                count_dict[second_num] = 1

    average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    result_list = [(average_dict[key], key) for key in average_dict]
    return result_list

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return list(e_x / e_x.sum(axis=0))

def return_eider_logits(rel2id_path = '../ass3/ATLOP/meta/rel2id.json',
                        doclogits_path = './dataNEW/res_slm_llm_edier/title2score_eider_EIDER_bert_eider_test_best.pkl',
                        rulelogits_path = './dataNEW/res_slm_llm_edier/title2score_evi_rule_EIDER_bert_eider_test_best.pkl'):

    rel2id = json.load(open(rel2id_path, 'r'))

    id2rel = {}
    for key in rel2id.keys():
        id2rel[rel2id[key]] = key
        
    all_res = pd.read_pickle(doclogits_path) 
    all_res2 = pd.read_pickle(rulelogits_path) 


    doc_relations = []

    for key in all_res.keys():
        doc_relation = {}
        for pair in all_res[key].keys():
            doc_topk = process_data(all_res[key][pair])
            relation_list = [-100] * 97
            NA_in = False
            min_score = 1000

            for topk in doc_topk:
                if topk[1] == 0:
                    NA_in = True
                    NA_score = topk[0]
                min_score = min(min_score, topk[0])

            if not NA_in:
                NA_score = min_score - 1

            for topk in doc_topk:    
                relation_list[topk[1]] = topk[0] - NA_score

            if key in all_res2 and pair in all_res2[key]:
                doc_topk = process_data(all_res2[key][pair])
                NA_in = False
                min_score = 1000
                for topk in doc_topk:
                    if topk[1] == 0:
                        NA_in = True
                        NA_score = topk[0]
                    min_score = min(min_score, topk[0])

                if not NA_in:
                    NA_score = min_score - 1
   
                for topk in doc_topk:  
                    if relation_list[topk[1]] == -100:
                        relation_list[topk[1]] = topk[0] - NA_score
                    else:
                        relation_list[topk[1]] += topk[0] - NA_score
                        
            for i in range(len(relation_list)):
                relation_list[i] = relation_list[i] - min(relation_list)
                        
            sorted_numbers = sorted(enumerate(relation_list), key=lambda x: x[1], reverse=True)

            top5_names = [id2rel[index] for index, _ in sorted_numbers]
            top5_values = [value for _, value in sorted_numbers]

            relations = {}
            for i in range(len(top5_names)):
                relations[top5_names[i]] = top5_values[i]

            doc_relation[pair] = relations
        doc_relations.append(doc_relation)

    return doc_relations