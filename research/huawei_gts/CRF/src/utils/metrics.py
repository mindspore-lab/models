"""metric util"""

def get_metric(P_ans, valid):
    predict_score = 0  # 预测正确个数
    predict_number = 0  # 预测结果个数
    total_number = 0  # 标签个数
    for i in range(len(P_ans)):
        predict_number += len(P_ans[i])
        total_number += len(valid.features[i].entity)
        pred_true = [x for x in valid.features[i].entity if x in P_ans[i]]
        predict_score += len(pred_true)
    P = predict_score / predict_number if predict_number > 0 else 0.
    R = predict_score / total_number if total_number > 0 else 0.
    f1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.
    print(f'f1 = {f1}， P(准确率) = {P}, R(召回率) = {R}')


if __name__ == '__main__':
    import mindspore.dataset as ds
    from dataset import read_data, get_dict, GetDatasetGenerator

    train = read_data('../../conll2003/train.txt')
    test = read_data('../../conll2003/test.txt')
    char_number_dict, id_indexs = get_dict(train[0])

    # 预测：test
    test = (test[0][:5], test[1][:5])
    test_dataset_generator = GetDatasetGenerator(test, id_indexs)
    dataset_test = ds.GeneratorDataset(test_dataset_generator, ["data", "length", "label"], shuffle=False)
    dataset_test = dataset_test.batch(batch_size=16)
    print("------------")
    print(dataset_test.get_dataset_size())

    v_pred = [[],
              [([2], 'LOC'), ([7], 'PER')],
              [([0, 1], 'PER')],
              [([0], 'LOC'), ([2, 3, 4], 'LOC')],
              [([0], 'LOC'), ([6, 7], 'MISC'), ([15], 'LOC')]
              ]
    print(get_metric(v_pred, test_dataset_generator))
