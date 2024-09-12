
from Seq2Seq import Seq2SeqSum
import mindspore
#from utils import make_word2id
#from CopySeq2Seq import CopySeq2SeqSum
# import time
# import eventlet
from train_seq2seqsum import src_vocab,tgt_vocab
import os
from nltk.translate.bleu_score import  sentence_bleu
from rouge import Rouge
from collections import defaultdict
from mindspore.amp import auto_mixed_precision

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
    # print(rouge_score[0]["rouge-1"])
    # print(rouge_score[0]["rouge-2"]) #ROUGE-2
    # print(rouge_score[0]["rouge-l"]) #ROUGE-L
    return rouge_score



# load model and use beam search
# to generate sentence summary
def beam_search(model_path, src_vocab,tgt_vocab,test,tgt,fout, beam_size=50):


    #load model
    model = Seq2SeqSum(len(src_vocab),len(tgt_vocab), 128, 256, 1)
    param_dict=mindspore.load_checkpoint(model_path)
    param_not_load,_=mindspore.load_param_into_net(model,param_dict)
    model = auto_mixed_precision(model, 'O2')#精度设置
    #model = Seq2SeqSum(len(src_vocab),len(tgt_vocab), 64, 128, 1)
    # model = CopySeq2SeqSum(len(word2id), 128, 256, 1)

    #ckpt = torch.load(model_path)['state_dict']
    #model.load_state_dict(ckpt)
    #breakpoint()
    tgt_idx={}
    for key in tgt_vocab:
        tgt_idx[tgt_vocab[key]]=key
    cnt = 0
    idx = 0
    tot,BLEU_score,Rouge_1,Rouge_2,Rouge_l=0,0,defaultdict(float),defaultdict(float),defaultdict(float)
    for test_src in test:
        cur_tgt=tgt[idx]
        idx += 1
        SENT = model._backbone.bs_decode(test_src, src_vocab,tgt_vocab, beam_size)
        cnt += 1
        if len(SENT)<1:
            pred=[]
        else:
            #print('cnt:%d gen:%d'%(cnt,len(SENT)))
            pred=[tgt_idx[x] for x in SENT[0][0]]
            pred=pred[1:-1]
        tgt_str=[tgt_idx[x] for x in cur_tgt]
        fout.write('%s\n%s\n' % (str(pred), str(tgt_str)))
        print('%d pred:%s \ntgt:%s\n' % (idx,str(pred), str(tgt_str)))

    #     BLEU_score+=bleu(pred,tgt)
    #     rouge_re=rouge_score(pred,tgt)[0]
    #     tot+=1
    #     for key in rouge_re['rouge-1']:
    #         Rouge_1[key]+=rouge_re['rouge-1'][key]
    #     for key in rouge_re['rouge-2']:
    #         Rouge_2[key]+=rouge_re['rouge-2'][key]
    #     for key in rouge_re['rouge-l']:
    #         Rouge_l[key]+=rouge_re['rouge-l'][key]
    #     print('-----------------------')
    #     print('pred:%s \ntgt:%s' % (str(pred), str(tgt)))
    #     print('%d BLEU:%s'%(idx,str(BLEU_score)))
    #     print('ROUGE-1 r:%s p:%s f:%s' % (str(Rouge_1['r'] ), str(Rouge_1['p']),str(Rouge_1['f'] )))
    #     print('ROUGE-2 r:%s p:%s f:%s' % (str(Rouge_2['r'] ), str(Rouge_2['p']), str(Rouge_2['f'] )))
    #     print('ROUGE-3 r:%s p:%s f:%s' % (str(Rouge_l['r']), str(Rouge_l['p']), str(Rouge_l['f'])))
    #     fout('-----------------------\n')
    #     fout.write('pred:%s \ntgt:%s\n' % (str(pred), str(tgt)))
    #     fout.write('%d BLEU:%s\n'%(idx,str(BLEU_score)))
    #     fout.write('ROUGE-1 r:%s p:%s f:%s\n' % (str(Rouge_1['r'] ), str(Rouge_1['p']),str(Rouge_1['f'] )))
    #     fout.write('ROUGE-2 r:%s p:%s f:%s\n' % (str(Rouge_2['r'] ), str(Rouge_2['p']), str(Rouge_2['f'] )))
    #     fout.write('ROUGE-3 r:%s p:%s f:%s\n' % (str(Rouge_l['r']), str(Rouge_l['p']), str(Rouge_l['f'])))
    #
    # fout2('tot:%d BLEU:%s'%(tot,str(BLEU_score/tot)))
    # fout2('ROUGE-1 r:%s p:%s f:%s' % (str(Rouge_1['r'] / tot), str(Rouge_1['p'] / tot),str(Rouge_1['f'] / tot)))
    # fout2('ROUGE-2 r:%s p:%s f:%s' % (str(Rouge_2['r'] / tot), str(Rouge_2['p'] / tot), str(Rouge_2['f'] / tot)))
    # fout2('ROUGE-3 r:%s p:%s f:%s' % (str(Rouge_l['r'] / tot), str(Rouge_l['p'] / tot), str(Rouge_l['f'] / tot)))


if __name__ == "__main__":

    # b1 = sentence_bleu([['this', 'is', 'a', 'fucking','test']], ['this', 'is' ,'a','test'],weights=(1, 0, 0, 0))
    # b2 = sentence_bleu([['this', 'is', 'a', 'fucking','test']], ['this', 'is' ,'a','test'],weights=(0, 1, 0, 0))
    # b3 = sentence_bleu([['this', 'is', 'a', 'fucking','test']], ['this', 'is' ,'a','test'],weights=(0, 0, 1, 0))
    #
    # rouge = Rouge()
    # rouge_score = rouge.get_scores(hyps=['this is a fucking test'], refs=['this is a test'])

    model_path = "./model_WMT14/ckpt-6e-0s.ckpt"
    file_1 = open('./WMT14/raw/src_test.txt', 'r')
    file_2 = open('./WMT14/raw/tgt_test.txt', 'r')
    out_path='./output/WMT14_output.txt'
    # out_path2 = './output/WMT14_re.txt'


    fout=open(out_path,'w',encoding='ISO-8859-1')

    #fout2=open(out_path2,'w')
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
