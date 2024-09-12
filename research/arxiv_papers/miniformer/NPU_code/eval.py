from nltk.translate.bleu_score import  sentence_bleu
from rouge import Rouge
from collections import defaultdict

def bleu(src, tgt):
    src = [str(x) for x in src]
    score1 = sentence_bleu([src], tgt, weights=(1, 0, 0, 0))
    score2 = sentence_bleu([src], tgt, weights=(0, 1, 0, 0))
    score3 = sentence_bleu([src], tgt, weights=(0, 0, 1, 0))
    score4 = sentence_bleu([src], tgt, weights=(0, 0, 0, 1))

    return score1, score2, score3, score4


def rouge_score(src, tgt):
    rouge = Rouge()
    src = ' '.join([str(x) for x in src])
    tgt = ' '.join(tgt)

    rouge_re = rouge.get_scores(hyps=src, refs=tgt)
    # print(rouge_re[0]["rouge-1"])
    # print(rouge_re[0]["rouge-2"]) #ROUGE-2
    # print(rouge_re[0]["rouge-l"]) #ROUGE-L
    return rouge_re

path='./output/WMT14_output.txt'
out_path='./output/WMT14_result.txt'
fin=open(path,'r',encoding='ISO-8859-1')
data=fin.readlines()
tot, bleu_socre, rouge_1, rouge_2, rouge_l = 0, 0, defaultdict(float), defaultdict(float), defaultdict(float)
bleu_socre2, bleu_socre3, bleu_socre4 = 0, 0, 0
ind=0
while ind<len(data):
    pred=eval(data[ind].rstrip('\r\n'))
    ind+=1
    tgt=eval(data[ind].rstrip('\r\n'))
    ind+=1
    if pred==[]:
        bleu_re=[0,0,0,0]
        rouge = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                         'rouge-2':{'f':0,'p':0,'r':0},'rouge-l':{'f':0,'p':0,'r':0}}
    else:
        bleu_re = bleu(pred, tgt)
        rouge = rouge_score(pred, tgt)[0]
    bleu_socre+=bleu_re[0]
    bleu_socre2+=bleu_re[1]
    bleu_socre3+=bleu_re[2]
    bleu_socre4+=bleu_re[3]
    for key in rouge['rouge-1']:
        rouge_1[key]+=rouge['rouge-1'][key]
    for key in rouge['rouge-2']:
        rouge_2[key]+=rouge['rouge-2'][key]
    for key in rouge['rouge-l']:
        rouge_l[key]+=rouge['rouge-l'][key]
    tot+=1
fout=open(out_path,'w')
fout.write('tot:%d BLEU:%s %s %s %s\n'%(tot,str(bleu_socre/tot),str(bleu_socre2/tot),str(bleu_socre3/tot),str(bleu_socre4/tot)))
fout.write('ROUGE-1 r:%s p:%s f:%s \n'%(rouge_1['r']/tot,rouge_1['p']/tot,rouge_1['f']/tot))
fout.write('ROUGE-2 r:%s p:%s f:%s \n'%(rouge_2['r']/tot,rouge_2['p']/tot,rouge_2['f']/tot))
fout.write('ROUGE-l r:%s p:%s f:%s \n'%(rouge_l['r']/tot,rouge_l['p']/tot,rouge_l['f']/tot))
