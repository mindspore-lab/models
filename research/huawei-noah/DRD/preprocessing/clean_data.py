import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_data(file_path, ignore_no_positive_query = True):
    input_file_path = file_path
    output_file_path = file_path + 'cleaned/'
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)
    datasets = ['train.txt', 'vali.txt', 'test.txt']
    for dataset in datasets:
        clean_oneset(input_file_path, output_file_path, dataset, ignore_no_positive_query)

def clean_oneset(input_file_path, output_file_path, set_name, ignore_no_positive_query):
    # Sort features by ID, count positive documents
    qid_list_data = {}
    qid_label_sum = {}

    feature_matrix = []
    qid_vec = []
    label_vec = []
    str_list = []
    with open(input_file_path + set_name) as fin:
        # Min MAX Scale(only use in WEB10K)
        # for line in fin:
        #     arr = line.strip().split(' ')
        #     label = int(arr[0])
        #     qid = int(arr[1].split(':')[1])
        #     if qid not in qid_list_data:
        #         qid_list_data[qid] = []
        #         qid_label_sum[qid] = 0
        #     feature_list = [x.split(':')[1] for x in arr[2:]]
        #     feature_matrix.append(feature_list)
        #     qid_vec.append(qid)
        #     label_vec.append(label)
        #     qid_label_sum[qid] += label

        # scale_tool = MinMaxScaler().fit(np.array(feature_matrix))
        # scaled_fe_mat = scale_tool.transform(feature_matrix)
        # for i in range(len(qid_vec)):
        #     fe_ls = scaled_fe_mat[i]
        #     fe_idx_ls = list(range(1,len(fe_ls)+1))
        #     # print('fe_ls: ',fe_ls)
        #     # print('zip: ', [label_vec[i], qid_vec[i]] + [ str(x[0])+':'+str(x[1]) for x in zip(fe_idx_ls, fe_ls) ])
        #     str_line = ''.join([str(label_vec[i])+' ', 'qid:'+str(qid_vec[i])+' '] + [ str(x[0])+':'+str(x[1])+' ' for x in zip(fe_idx_ls, fe_ls) ])
        #     # print(qid_vec[i])
        #     # print(str_line)
        #     qid_list_data[qid_vec[i]].append(str_line)


        for line in fin:
            arr = line.strip().split(' ')
            label = int(arr[0])
            qid = int(arr[1].split(':')[1])
            feature_list = arr[2:]
            idx_feature_map = {int(x.split(':')[0]): x for x in feature_list if ':' in x}
            sorted_idx_feature_list = sorted(
                idx_feature_map.items(), key=lambda k: k[0])
            if qid not in qid_list_data:
                qid_list_data[qid] = []
                qid_label_sum[qid] = 0
            qid_list_data[qid].append(
                ' '.join([arr[0], arr[1]] + [x[1] for x in sorted_idx_feature_list]))
            qid_label_sum[qid] += label

    with open(output_file_path + set_name, 'w') as fout:
        sorted_qid_lists = sorted(qid_list_data.items(), key=lambda k: k[0])
        for qid_list in sorted_qid_lists:
            if ignore_no_positive_query and qid_label_sum[qid_list[0]] < 1:
                continue
            for line in qid_list[1]:
                fout.write(line)
                fout.write('\n')

if __name__ == '__main__':
    #pass
    DATA_PATH = sys.argv[1]
    clean_data(DATA_PATH)