import json
import sys
import pandas as pd
import numpy as np 
from itertools import starmap

def load_json(file_path):
    load_dict_list = []
    with open(file_path, "r") as f:
         load_dict_list = json.load(f)
    data_frame = pd.DataFrame(load_dict_list)
    return data_frame

def select_rank_list_forOnequery(data_frame, queryID):
    selected_data_frame = data_frame[data_frame["queryID"] == queryID]
    key_and_value_dict = selected_data_frame.to_dict(orient='index')
    onequery_list = []
    for _, value in key_and_value_dict.items():
        onequery_list.append(value)
    return onequery_list

def get_label_list(query_list):
    label_list = [0 for i in range(len(query_list))]
    for doc in query_list:
        label_list[doc["rankPosition"]] = doc["label"]
    return label_list

def load_data_forEstimate(file_path):
    # load json file
    data_frame = load_json(file_path)

    # get max length of rank list
    df_count = data_frame.groupby('queryID').count()
    df_count.rename(columns={'docID':'doc_num'}, inplace = True)
    rank_list_size = df_count['doc_num'].max()

    # get unique query ids
    queryid_list = data_frame['queryID'].apply(int).values.tolist()
    queryid_list = list(set(queryid_list))

    # get query-doc pair for each query
    query_lists = list(starmap(select_rank_list_forOnequery, [(data_frame, qid) for qid in queryid_list]))
    label_lists = list(map(get_label_list, query_lists))
    query_lists = list(zip(queryid_list, query_lists))

    rank_data = RankData(query_lists, label_lists, rank_list_size)
    return rank_data


class RankData:
    def __init__(self, query_lists, label_lists, rank_list_size):
        self.query_lists = query_lists
        self.label_lists = label_lists
        self.rank_list_size = rank_list_size
        self.qid_lists = [ql[0] for ql in self.query_lists]
        self.query_num = len(query_lists)

        # zipf distribution param
        # self.alpha = 0.90820727
        # self.C = 0.05276632
        self.alpha = 0.74191474
        self.C = 0.01091755327

        rank_list_num = []
        data_size = 0
        for q in query_lists:
            rank_list_num.append(len(q[1]))
            data_size += len(q[1])
        self.data_size = data_size
        rank_order = pd.Series(rank_list_num).rank(method='average').values
        self.query_freq = [self.C / pow(r, self.alpha) for r in rank_order]
        # scale query_freq into a prob distribution
        self.query_prob = np.array(self.query_freq) * (1/sum(self.query_freq))
        last_one =1.0 -  sum(self.query_prob[:-1])
        self.query_prob[-1] = last_one

        

if __name__ == "__main__":
    FILE_PATH = sys.argv[1] + 'Train.json'
    #QUERY_ID = sys.argv[2]

    rank_data = load_data_forEstimate(FILE_PATH)



    
