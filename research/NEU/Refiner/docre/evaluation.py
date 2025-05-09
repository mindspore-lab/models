#!/usr/bin/env python
import sys
import os
import os.path
import json

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))
    
    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train
    
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train

def evaluate(data_path = "./docred", 
             test_data = "dev.json", 
             result_data="./result.json", 
             output_path="./", 
             train_annotated_path = "/train_annotated.json", 
             compare_distant = True):
    input_dir = data_path
    truth_dir = os.path.join(input_dir, 'ref')
    
    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    if os.path.isdir(truth_dir):
        fact_in_train_annotated = gen_train_facts(data_path + train_annotated_path, truth_dir)
        if compare_distant:
            fact_in_train_distant = gen_train_facts(data_path + "/train_distant.json", truth_dir)
        else:
            fact_in_train_distant = set([])

        output_filename = os.path.join(output_path, 'socres.txt')
        output_file = open(output_filename, 'w')

        truth_file = os.path.join(data_path, test_data)
        truth = json.load(open(truth_file))

        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']

                h_idx = label['h']
                t_idx = label['t']
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        tot_relations = len(std)

#         submission_answer_file = os.path.join(result_path, "result.json")
        submission_answer_file = result_data
        tmp = json.load(open(submission_answer_file))
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i-1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])
    #         else:
    #             print("remove", x['title'], x['h_idx'], x['t_idx'], x['r'])

        correct_re = 0
        correct_evidence = 0 
        pred_evi = 0 

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])

    
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            
            titleset2.add(title)
            if title not in title2vectexSet:
                continue

            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])

            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

        re_p = 1.0 * correct_re / len(submission_answer)

        re_r = 1.0 * correct_re / tot_relations
        if re_p+re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi>0 else 0
        evi_r = 1.0 * correct_evidence / (tot_evidences+0.000000000000000001)
        if evi_p+evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
        re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) / (len(submission_answer)-correct_in_train_distant)

        if re_p_ignore_train_annotated+re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train+re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)


        print("Precision:",re_p)
        print("Recall:",re_r)
        print ('RE_F1:', re_f1)
        print ('Evi_F1:', evi_f1)
        print ('RE_ign_F1:', re_f1_ignore_train_annotated)
        print ('RE_ignore_distant_F1:', re_f1_ignore_train)
        

        output_file.write("Precision: %f\n" % re_p)
        output_file.write("Recall: %f\n" % re_r)

        output_file.write("RE_F1: %f\n" % re_f1)
        output_file.write("Evi_F1: %f\n" % evi_f1)

        output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
        output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)


        output_file.close()