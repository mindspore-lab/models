from Seq2Seq import MiniFormer
import mindspore
from train_MiniFormer import src_vocab,tgt_vocab
import os
from nltk.translate.bleu_score import  sentence_bleu
from rouge import Rouge
from collections import defaultdict
if not os.path.exists('./output/'):
    os.mkdir('./output/')
def bleu(src,tgt):
    if src==[[]]:
        return 0
    score = sentence_bleu(src, tgt)
    return score
def rouge_score(src,tgt):
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=src, refs=tgt)
    return rouge_score
def beam_search(model_path, src_vocab,tgt_vocab,test,tgt,fout, beam_size=50):
    model = MiniFormer(len(src_vocab),len(tgt_vocab), 128, 256, 1)
    param_dict=mindspore.load_checkpoint(model_path)
    param_not_load,_=mindspore.load_param_into_net(model,param_dict)
    tgt_idx={}
    for key in tgt_vocab:
        tgt_idx[tgt_vocab[key]]=key
    cnt = 0
    idx = 0
    tot,BLEU_score,Rouge_1,Rouge_2,Rouge_l=0,0,defaultdict(float),defaultdict(float),defaultdict(float)
    for test_src in test:
        cur_tgt=tgt[idx]
        idx += 1
        SENT = model.bs_decode(test_src, src_vocab,tgt_vocab, beam_size)
        cnt += 1
        if len(SENT)<1:
            pred=[]
        else:
            pred=[tgt_idx[x] for x in SENT[0][0]]
            pred=pred[1:-1]
        tgt_str=[tgt_idx[x] for x in cur_tgt]
        fout.write('%s\n%s\n' % (str(pred), str(tgt_str)))
        print('%d pred:%s \ntgt:%s\n' % (idx,str(pred), str(tgt_str)))
if __name__ == "__main__":
    model_path = "./model_WMT14/ckpt-6e-0s.ckpt"
    file_1 = open('./WMT14/raw/src_test.txt', 'r')
    file_2 = open('./WMT14/raw/tgt_test.txt', 'r')
    out_path='./output/WMT14_output.txt'
    fout=open(out_path,'w',encoding='ISO-8859-1')
    test = []
    while True:
        line = file_1.readline().rstrip('\r\n')
        if not line:
            break
        line = [int(x) for x in line.split(' ')]
        test.append(line)
    file_1.close()
    tgt=[]
    while True:
        line = file_2.readline().rstrip('\r\n')
        if not line:
            break
        line = [int(x) for x in line.split(' ')]
        tgt.append(line)
    file_2.close()
    beam_search(model_path, src_vocab,tgt_vocab,test[:],tgt,fout)