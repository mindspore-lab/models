import numpy
import json
import pandas as pd
import re
import sys
#import jsonlines
import os
sys.path.append("..")
from utils.scale import scale_dataset
dict_from_query_to_id = {}
# dict_from_doc_to_id = {}

def read_score(file_name):
    score_list = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            score = float(line)
            score_list.append(score)
    return score_list

def read_dataset(file_name, data_type):
    each_row = []
    query_id = 0
    doc_id = 0
    is_zero = 0
    is_one = 0
    is_two = 0
    is_three = 0
    is_four = 0
    dataset_name = file_name.split('/')[-3]
    with open(file_name, "r") as f:
        for line in f:
            query_doc_dict = {}
            has_query_id = False
            # has_doc_id = False
            has_label = False
            feature = []
            query_doc_dict["feature"] = feature
            line = line.replace("\n", "")
            items = line.split(" ")
            for temp in items:
                if not has_label:
                    query_doc_dict["label"] = int(temp[0])
                    has_label = True
                    query_doc_dict["docID"] = doc_id
                    doc_id += 1
                    if query_doc_dict["label"] == 0:
                        is_zero += 1
                    elif query_doc_dict["label"] == 1:
                        is_one += 1
                        # if dataset_name == 'YaHooC14B' or dataset_name == 'WEB10K':
                        #     query_doc_dict['label'] = 0
                    elif query_doc_dict["label"] == 2:
                        is_two += 1
                        # query_doc_dict["label"] = 1
                        # if dataset_name == 'YaHooC14B':
                        #     query_doc_dict['label'] = 0
                        # if dataset_name == 'WEB10K':
                        #     query_doc_dict['label'] = 0
                    elif query_doc_dict["label"] == 3:
                        is_three += 1
                        # query_doc_dict["label"] = 1
                    else :
                        is_four += 1
                        # query_doc_dict["label"] = 1
                elif not has_query_id and "qid" in temp:
                    qid = temp.split(":")
                    has_query_id = True
                    qid = qid[1]
                    query_doc_dict["oriQueryID"] = qid
                    if qid in dict_from_query_to_id:
                        query_doc_dict["queryID"] = dict_from_query_to_id[qid]
                    else:
                        query_doc_dict["queryID"] = query_id
                        dict_from_query_to_id[qid] = query_id
                        query_id = query_id + 1
                # elif not has_doc_id and "docid" in temp:
                #     doc = re.findall("#.*?(\w+?-\w+?-\w+)", line)
                #     has_doc_id = True
                #     doc = doc[0]
                #     query_doc_dict["oriDocID"] = doc
                #     if doc in dict_from_doc_to_id:
                #         query_doc_dict["docID"] = dict_from_doc_to_id[doc]
                #
                #     else:
                #         query_doc_dict["docID"] = doc_id
                #         dict_from_doc_to_id[doc] = doc_id
                #         doc_id = doc_id + 1
                elif ":" in temp:
                    if ('YaHooC14B' not in dataset_name):
                        featrue_number = float(temp.split(":")[1])
                        query_doc_dict["feature"].append(featrue_number)
            
            if ('YaHooC14B' in dataset_name) and (data_type == 'Train'):
                # raw_feature = [list(map(float,fe.split(':')))  for fe in items[2:]]
                # query_doc_dict["feature"] = raw_feature
                raw_feature = {fe.split(':')[0]:fe for fe in items[2:]}
                for idx in range(700):
                    cur_id = idx + 1
                    if str(cur_id) in raw_feature.keys():
                        query_doc_dict["feature"].append(float(raw_feature[str(cur_id)].split(':')[1]))
                    else:
                        query_doc_dict["feature"].append(0.0)  
            elif ('YaHooC14B' in dataset_name) and (data_type != 'Train'):
                raw_feature = {fe.split(':')[0]:fe for fe in items[2:]}
                for idx in range(700):
                    cur_id = idx + 1
                    if str(cur_id) in raw_feature.keys():
                        query_doc_dict["feature"].append(float(raw_feature[str(cur_id)].split(':')[1]))
                    else:
                        query_doc_dict["feature"].append(0.0)   

            each_row.append(query_doc_dict)
    print(file_name)
    print("0:" + str(is_zero))
    print("1:" + str(is_one))
    print("2:" + str(is_two))
    print("3:" + str(is_three))
    print("4:" + str(is_four))
    print()
    return each_row

def set_rank_score(data_dict_list, score_list):
    i = 0
    for each_dict in data_dict_list:
        each_dict["rankScore"] = score_list[i]
        i += 1
    # print("set_rank_score")
    # print(i)
    # print(len(data_dict_list))
    return data_dict_list

def set_rank_position(data_dict_list):
    sorted_dataframe = []
    data_frame = pd.DataFrame(data_dict_list)
    data_group = data_frame.groupby("queryID", sort=False)
    for _, group in data_group:
        # print(k1)
        # print(group)
        group = group.sort_values("rankScore", ascending=False)
        rank_postion = range(group.shape[0])
        group["rankPosition"] = rank_postion
        sorted_dataframe.append(group)
    return sorted_dataframe

def write_to_file(sorted_dataframe, out_path, train_or_test):
    if train_or_test:
        result_list = []
        for temp in sorted_dataframe:
            key_and_value_dict = temp.to_dict(orient='index')
            for _, value in key_and_value_dict.items():
                result_list.append(value)

        # with jsonlines.open(out_path, mode='w') as writer:
        #     for temp in result_list:
        #         writer.write(temp)
        with open(out_path, "w") as f:
            json.dump(result_list, f)

    else:
        # with jsonlines.open(out_path, mode='w') as writer:
        #     for temp in sorted_dataframe:
        #         writer.write(temp)
        with open(out_path, "w") as f:
            json.dump(sorted_dataframe, f)

def handle_data(file_path):
    file_list = ["train", "vali"]
    out_file_path = file_path + 'json_file/'

    predict_file_path = file_path + 'predict/'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    out_file_list = ["Train", "Vali"]

    for i in range(len(file_list)):
        data_path = file_path+ 'cleaned/' + file_list[i] + ".txt"
        data_predict_path = predict_file_path + file_list[i] + "_predict.txt"
        data_out_file_path = out_file_path + out_file_list[i] + ".json"

        data = read_dataset(data_path, out_file_list[i])
        predict_data = read_score(data_predict_path)
        train_data_with_score = set_rank_score(data, predict_data)
        sorted_dataframe = set_rank_position(train_data_with_score)
        write_to_file(sorted_dataframe, data_out_file_path, True)

    test_path = file_path + 'cleaned/' + "test.txt"
    test_out_file_path = out_file_path + "Test.json"
    test_data = read_dataset(test_path, 'Test')
    write_to_file(test_data, test_out_file_path, False)

if __name__ == "__main__":
    DATASET_PATH = sys.argv[1]
    handle_data(DATASET_PATH)




