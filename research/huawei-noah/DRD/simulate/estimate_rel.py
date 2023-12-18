import numpy as np
import random
import json
import copy
import os
import sys
sys.path.append("..")
import utils.click_model as CM
from utils.load_data import load_data_forEstimate
from tqdm import tqdm


class RandomizedRelevanceEstimator:
    def __init__(self):
        pass 

    def estimateRelevanceFromModel(self, click_model, rank_data, session_num):
        self.click_model = click_model
        # simulate uniform data use 1% of session_num
        qid_log_map = {}
        for _ in tqdm(range(int(session_num*0.01))):
            # sample query with zipf distribution
            index = np.random.choice(rank_data.qid_lists, 1, p=rank_data.query_prob, replace=True)[0]
            # sample query with uniform distribution
            #index = random.randint(0,len(rank_data.label_lists)-1)
            dids = [x for x in range(len(rank_data.label_lists[index]))]
            label_list = list(zip(dids, rank_data.label_lists[index]))
            if index not in qid_log_map.keys():
                qid_log_map[index] = np.zeros((len(label_list), len(label_list)))
            
            np.random.shuffle(label_list)
            dids_aftershuffle = [d[0] for d in label_list]
            label_list_withoutdid = [d[1] for d in label_list]
            click_list, _, _, _ = self.click_model.sampleClicksForOneList(label_list_withoutdid)
            for posi_idx in range(len(click_list)):
                qid_log_map[index][dids_aftershuffle[posi_idx]][posi_idx] += click_list[posi_idx]

        # estimate relevance of each query list
        qid_rel_map = {}
        for qid, log_mat in qid_log_map.items():
            estimate_rel = [sum(doc_log)/(session_num/len(label_list)) for doc_log in log_mat]
            max_val = max(estimate_rel)
            min_val = min(estimate_rel)
            if (max_val-min_val) == 0:
                estimate_scale = [0 for rel in estimate_rel]
            else:
                estimate_scale = [(rel-min_val)/(max_val-min_val) for rel in estimate_rel]
            qid_rel_map[qid] = estimate_scale

        self.estimate_rel = qid_rel_map

    def outputResultToData(self, rank_data):
        query_lists_for_train = []
        for query_list in rank_data.query_lists:
            if query_list[0] in self.estimate_rel.keys():
                for idx, rel in enumerate(self.estimate_rel[query_list[0]]):
                    query_list[1][idx]['estimate_label'] = rel
                query_lists_for_train.extend(query_list[1])
        return query_lists_for_train

if __name__ =="__main__":
    pbm = CM.PositionBiasedModel()
    pbm.setClickProb(0.1, 1.0, 1)
    pbm.setExamProb(1)

    rank_data = load_data_forEstimate('../../datasets/MQ2008/json_file/Vali.json')
    #label_lists = [[1,1,0,1,0,0,1,1,0,0,1,0]]
    session_num = 1e6

    estimator = RandomizedRelevanceEstimator()
    estimator.estimateRelevanceFromModel(pbm, rank_data, session_num)
    rank_data = estimator.outputResultToData(rank_data)

    #print(rank_data.query_lists[106])
    json_dict = []
    for ql in rank_data.query_lists:
        json_dict.extend(ql[1])

    with open('../../datasets/MQ2008/json_file/Vali_estimate.json', 'w') as fout:
        #for query_list in rank_data.query_lists:
        json.dump(json_dict, fout)

    #print(qid_rel_map[106])
    #print(rank_data.label_lists[106])






