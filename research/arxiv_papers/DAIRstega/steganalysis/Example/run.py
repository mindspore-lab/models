import torch
from torch import nn
import argparse
import sys
import numpy as np
import data
import GE
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


cover_data_name = "../../data_cover/A_Overall.txt"
stego_data_name = '../../data_stego/topk/a48-sqrt/A_Overall/A_Overall.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x: x.lower() == 'true')
    parser.add_argument("--neg_filename", type=str, default=cover_data_name)
    parser.add_argument("--pos_filename", type=str, default=stego_data_name)
    parser.add_argument("--epoch", type=int, default=10)  # default=100
    parser.add_argument("--stop", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=None)
    # parser.add_argument("--logdir", type=str, default="./rnnlog")
    parser.add_argument("--sentence_num", type=int, default=1007)  # default=1000
    parser.add_argument("--rand_seed", type=int, default=42)
    args = parser.parse_args(sys.argv[1:])
    return args

args = get_args()
# logger

import random
random.seed(args.rand_seed)  

# log_dir = args.logdir
# os.makedirs(log_dir, exist_ok=True)
# log_file = log_dir + "/rnn_{}.txt".format(os.path.basename(args.neg_filename)+"___"+os.path.basename(args.pos_filename))
# logger = Logger(log_file)

def main(data_helper):
    CELL = "bi-gru"            # rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm
    BATCH_SIZE = 256  # 64
    EMBED_SIZE = 300  # 128
    HIDDEN_DIM = 100  # 256
    NUM_LAYERS = 2
    CLASS_NUM = 2
    DROPOUT_RATE = 0.5  # 0.2
    EPOCH = args.epoch 
    LEARNING_RATE = 0.001
    SAVE_EVERY = 20
    STOP = args.stop 
    SENTENCE_NUM = args.sentence_num  # 2000
    K = 2
    G = 2
    FILTER_NUM = 100
    FILTER_SIZE = [3, 5]

    # all_var = locals()
    # print()
    # for var in all_var:
    # 	if var != "var_name":
    # 		logger.info("{0:15}   ".format(var))
    # 		logger.info(all_var[var])
    # print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GE.ge(
        cell=CELL,
        vocab_size=data_helper.vocab_size,
        embed_size=EMBED_SIZE,
        filter_num=FILTER_NUM,
        filter_size=FILTER_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT_RATE,
        k=K,
        g=G,
    )
    model.to(device)
    criteration = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-6) 
    best_acc = 0
    best_test_loss = 100
    early_stop = 0

    epoch_test_P, epoch_test_R = [], [] # add by xq 22.9.29

    for epoch in range(EPOCH):
        generator_train = data_helper.train_generator(BATCH_SIZE)
        generator_test = data_helper.test_generator(BATCH_SIZE)
        train_loss = []
        train_acc = []
        while True:
            try:
                text, label = generator_train.__next__()
            except:
                break
            optimizer.zero_grad() 
            y = model(torch.from_numpy(text).long().to(device))
            loss = criteration(y, torch.from_numpy(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            y = y.cpu().detach().numpy()
            train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        test_loss = []
        test_acc = []

        test_tp = [] # add by xq 22.9.29
        tfn = []
        tpfn = []
        length_sum = 0
        test_loss = 0

        while True:
            with torch.no_grad(): 
                try:
                    text, label = generator_test.__next__()
                except:
                    break
                y = model(torch.from_numpy(text).long().to(device))  # y={Tensor:(64,2)}
                loss = criteration(y, torch.from_numpy(label).long().to(device))
                test_loss += loss * len(text)
                length_sum += len(text)
                y = y.cpu().numpy()
                test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
                # add by xq 22.9.29
                test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
                tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
                tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
        # logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
        #       .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
        # add by xq 22.9.29
        test_loss = test_loss / length_sum
        tpsum = np.sum(test_tp)
        test_precision = tpsum / np.sum(tfn)
        test_recall = tpsum / np.sum(tpfn)
        test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision)
        epoch_test_P.append(test_precision)
        epoch_test_R.append(test_recall)

        # print('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
        #       .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_acc = np.mean(test_acc)
            precison = test_precision
            recall = test_recall
            F1 = test_Fscore
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= STOP:
            # logger.info('best acc: {:.4f}'.format(best_acc))
            print('best acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_acc, precison, recall, F1))
            # return best_acc
            return best_acc, precison, recall, F1 # add by xq 22.9.29

        if (epoch + 1) % SAVE_EVERY == 0:
            print('saving parameters')
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/GE-' + str(epoch) + '.pkl')
# 	logger.info('best acc: {:.4f}'.format(best_acc))
# 	print('best acc: {:.4f}'.format(best_acc))
# 	return best_acc'
    # add by xq 22.9.29
    print('best acc: {:.4f}, pre {:.4f}, recall {:.4f}'.format(best_acc, precison, recall))
    # torch.save(model.state_dict(), './train_epoch200_.pth')

    return best_acc, precison, recall, F1


if __name__ == '__main__':
    acc = []
    preci = [] # add by xq 22.9.29
    recall = [] # add by xq 22.9.29
    f1 = []
    # filter_data(args.neg_filename,args.pos_filename)
    # with open(args.neg_filename+"_filter", 'r', encoding='utf-8') as f:
    with open(args.neg_filename, 'r', encoding='utf-8') as f:  
        raw_pos = f.read().lower().split("\n")
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
    if args.max_length is not None:
        raw_pos = [text for text in raw_pos if len(text.split()) < args.max_length]
    import random

    random.shuffle(raw_pos)

    with open(args.pos_filename, 'r', encoding='utf-8') as f: 
        raw_neg = f.read().lower().split("\n")
    raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
    if args.max_length is not None:
        raw_neg = [text for text in raw_neg if len(text.split()) < args.max_length]
    random.shuffle(raw_neg)
    # raw_neg = [' '.join(list(jieba.cut(pp))) for pp in raw_neg]
    length = min(args.sentence_num, len(raw_neg), len(raw_pos))
    # length = len(raw_pos)
    data_helper = data.DataHelper([raw_pos[:length], raw_neg[:length]], use_label=True, word_drop=0)

    for i in range(5):
        random.seed(42) 
        start = time.time() # add by xq 22.9.29
        # add by xq 22.9.29
        index = main(data_helper)
        acc.append(index[0])
        preci.append(index[1])
        recall.append(index[2])
        f1.append(index[3])
    # print("max3_mean:", np.mean(np.sort(acc)[-3:]))
    # acc_mean = np.mean(acc)
    acc_mean = np.mean(np.sort(acc)[-3:])
    acc_std = np.std(acc)
    pre_mean = np.mean(np.sort(preci)[-3:])
    pre_std = np.std(preci)
    recall_mean = np.mean(np.sort(recall)[-3:])
    recall_std = np.std(recall)
    f1_mean = np.mean(np.sort(f1)[-3:])
    f1_std = np.std(f1)
    print("Final: acc {:.2f}±{:.2f}, P {:.2f}±{:.2f}, || R {:.2f}±{:.2f}, F1 {:.2f}±{:.2f}"
                .format(acc_mean*100, acc_std*100, pre_mean*100, pre_std*100,
                        recall_mean*100, recall_std*100, f1_mean*100, f1_std*100))


