import os
import json
import pandas as pd
import re


def remove_space_before_punctuation(text):
    cleaned_text = re.sub(r'\s+([.,\'\"])', r'\1', text)
    cleaned_text = re.sub(r'\(\s+', '(', cleaned_text)
    cleaned_text = re.sub(r'\s+\)', ')', cleaned_text)
    cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)

    return cleaned_text


def return_rel2dict(file_path = './dataset/docred/rel_info.json'):
    fr = open(file_path, 'r', encoding='utf-8')
    rel_info = fr.read()
    rel_info = eval(rel_info)

    p_to_num = {}
    num_to_p = {}
    for i,key in enumerate(rel_info.keys()):
        p_to_num[key] = i
        num_to_p[i] = key
    num_to_p[len(rel_info.keys())] = 'NA'
    p_to_num['NA'] = len(rel_info.keys())

    p_to_name = {}
    name_to_p = {}
    for key in rel_info.keys():
        p_to_name[key] = rel_info[key]
        name_to_p[rel_info[key]] = key
    p_to_name['NA'] = 'NA'
    name_to_p['NA'] = 'NA'
    return p_to_num, num_to_p, p_to_name, name_to_p

def return_templates(file_path = './dataNEW/rel_templates.xlsx'):
    df_templates = pd.read_excel(file_path)

    p2templates = {}
    ps = df_templates['relation ID'].values
    templates = df_templates['relation template'].values
    for i,p in enumerate(ps):
        p2templates[p.strip()] = templates[i]
    
    return p2templates

def return_docred(file_path = './dataset/docred/dev.json',test_data=False): 
    fr = open(file_path, 'r', encoding='utf-8')
    json_info = fr.read()
    df = pd.read_json(json_info)

    titles = []
    for i in range(len(df['vertexSet'])):
        titles.append(df['title'][i])

    entities = []
    for i in range(len(df['vertexSet'])):
        enames = []
        for entity_class in df['vertexSet'][i]:
            ename = set()
            for entity_name in entity_class:
                ename.add(entity_name['name'])
            enames.append(list(ename))
        entities.append(enames)
        
    entity_types = []
    for i in range(len(df['vertexSet'])):
        etypes = []
        for entity_class in df['vertexSet'][i]:
            entity_type = set()
            for entity_name in entity_class:
                entity_type.add(entity_name['type'])
            etypes.append(list(entity_type)[0])
        entity_types.append(etypes)
        
    entity_indexs = []
    for i in range(len(df['vertexSet'])):
        eindexs = []
        for entity_class in df['vertexSet'][i]:
            eindex = set()
            for entity_name in entity_class:
                eindex.add(entity_name['sent_id'])
            eindexs.append(list(eindex))
        entity_indexs.append(eindexs)
        
    documents_raw = []
    for i in range(len(df['sents'])):
        document_raw = []
        for j,sentence in enumerate(df['sents'][i]):
            sentence_str = ""
            for word in sentence[:-1]:
                sentence_str += word
                sentence_str += " "
            sentence_str += sentence[-1]
            document_raw.append(remove_space_before_punctuation(sentence_str))
        documents_raw.append(document_raw) 
        
    relations = []
    if test_data == False:
        for i in range(len(df['sents'])):
            relation = df['labels'][i]
            relations.append(relation)
    else:
        relations = []    
    return titles, entities, entity_types, entity_indexs, documents_raw, relations 




