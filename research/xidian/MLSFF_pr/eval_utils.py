#coding:utf-8

import mindspore as ms
import numpy as np
import json
import os
import tools.utils as utils
import sys
sys.path.append('./')
from settings import for_eval_json
from coco_caption_3.pycocotools.coco import COCO
from coco_caption_3.pycocoevalcap.eval import COCOEvalCap

def language_eval(args,preds, model_id, split):
    annFile =for_eval_json 
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', args.pretrained_model_path + '_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) 
    cocoRes = coco.loadRes(preds_filt)                                        
    cocoEval = COCOEvalCap(coco, cocoRes)                                     
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out

def eval_split(args,model, crit, loader, eval_kwargs):
    num_images = eval_kwargs['val_images_use']
    split = eval_kwargs['split']
    model.set_train(mode=False)
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    tem=0
    while True:
        data = loader.get_batch(split)        
        n = n + loader.batch_size
        
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['attr_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [ms.Tensor(_) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, attr_feats = tmp

        seq = model.sample(fc_feats, att_feats, attr_feats, eval_kwargs)[0]
        seq = ms.ops.stop_gradient(seq)
        sents = utils.decode_sequence(loader.ix_to_word, seq.asnumpy())

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 1) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()                                                 
            
        if ix0-tem>500:
            tem=ix0
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    lang_stats = language_eval(args,predictions, eval_kwargs['net'], split)
    model.set_train(mode=True)
    return loss_sum/loss_evals, predictions, lang_stats
