import os
import re
import pandas as pd
import pickle
import mindspore as ms
import numpy as np
import os
import clip
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig

from stable_diffusion_v2.stable_diffusion import SDPipeline

DATA_PATH_POG = './data/raw_data/POG'
DATA_PATH_MOVIE = './data/raw_data/ml-latest-small'
HIS_SEQ_LEN = {'POG': 5, 'movie': 3}

def initEnv():
    # set model config
    llama_config = MindFormerConfig("./config/run_llama2_7b_910b.yaml")

    # 初始化环境
    init_context(use_parallel=llama_config.use_parallel,
                    context_config=llama_config.context,
                    parallel_config=llama_config.parallel)
    return llama_config

def loadCLIP():
    model, preprocess = clip.load("./checkpoint_download/clip/ViT_B_32.ckpt", device="Ascend")
    return model, preprocess

def loadLlama(config):
    model_config = LlamaConfig(**config.model.model_config)
    # model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.use_past = True
    model_config.seq_length = 512
    model_config.checkpoint_name_or_path = "./checkpoint_download/llama2/llama2_7b-chat.ckpt"
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llama2_7b")

    model = LlamaForCausalLM(model_config)
    model.set_train(False)
    return tokenizer, model
    
def loadSDPipeline(dataset):
    if dataset == 'POG':
        raise NotImplementedError
    elif dataset == 'movie':
        sd_pipeline = SDPipeline()
    elif dataset == 'emoticon':
        raise NotImplementedError
    return sd_pipeline

def loadPOGTestData():
    test_data_path = os.path.join(DATA_PATH_POG, 'my_valid_data.txt')
    item_data_path = os.path.join(DATA_PATH_POG, 'item_data.txt')
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_case_list = []
        lines = f.readlines()
        for line in lines:
            item_list = line.split('\n')[0].split(';')
            if len(item_list) > 1:
                test_case_list.append({'history': item_list})
        
        
    with open(item_data_path, 'r', encoding='utf-8') as f:
        item_data = {}
        lines = f.readlines()
        for line in lines:
            #print(line.split('\n')[0])
            item_id, cate_id, pic_url, title = line.split('\n')[0].split(',')[:4]
            if pic_url[:5] != 'http:' and pic_url[:6] != 'https:':
                pic_url = 'http:' + pic_url
            item_data[item_id] = {
                'cate_id': cate_id,
                'pic_url': pic_url,
                'title': title
            }
            
    return test_case_list, item_data

def loadMovieTestData():
    keyword_data_path = os.path.join(DATA_PATH_MOVIE, 'keywords_in_id.pkl')
    item_data_path = os.path.join(DATA_PATH_MOVIE, 'info.csv')
    test_case_path = os.path.join(DATA_PATH_MOVIE, 'my_valid_data.txt')
    
    with open(keyword_data_path, 'rb') as f:
        keywords_in_id = pickle.load(f)
    
    item_data = {}
    mv_info = pd.read_csv(item_data_path)
    for index, row in mv_info.iterrows():
        only_name = re.findall(r'^(.*) \(\d+\) *$', row['name'])
        if len(only_name):
            row['name'] = only_name[0]
        movie_id = str(row['id'])
        item_data[movie_id] = dict(row)
        item_data[movie_id]['keyword'] = keywords_in_id[row['id']]
        


    with open(test_case_path, 'r', encoding='utf-8') as f:
        test_case_list = []
        lines = f.readlines()
        for line in lines:
            item_list, target_id = line.split('\n')[0].split(',')
            test_case_list.append({'history': item_list.split(';'), 'target':target_id})
    return test_case_list, item_data

def loadEmoticonTestData():
    pass

def loadTestData(dataset='POG'):
    if dataset == 'POG':
        return loadPOGTestData()
    if dataset == 'movie':
        return loadMovieTestData()
    

def loadPOGFullData(his_seq_len=HIS_SEQ_LEN['POG'], num_user_data=1000):
    user_data_path = os.path.join(DATA_PATH_POG, 'user_data.txt')
    item_data_path = os.path.join(DATA_PATH_POG, 'item_data.txt')
    outfit_data_path = os.path.join(DATA_PATH_POG, 'outfit_data.txt')
    image_data_path = os.path.join(DATA_PATH_POG, 'image_data_1000_io.pkl')

    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
        image_data_dict = {}
        for _, item_id, image in image_data:
            image_data_dict[item_id] = image
            
    with open(outfit_data_path, 'r', encoding='utf-8') as f:
        outfit_data = {}
        lines = f.readlines()
        for line in lines:
            outfit_id, item_list = line.split('\n')[0].split(',')
            outfit_data[outfit_id] = item_list.split(';')
            
    with open(user_data_path, 'r', encoding='utf-8') as f:
        user_data = []
        for i in range(num_user_data):
            data = f.readline()
            user_id, item_list, outfit_id = data.split('\n')[0].split(',')
            item_list = [x.strip() for x in item_list.split(';')][-his_seq_len:]
            item_list = [x for x in item_list if x in image_data_dict]
            target_list = [x for x in outfit_data[outfit_id] if x in image_data_dict]
            if len(item_list) == 0 or len(target_list) == 0:
                continue
            user_data.append({
                'history': item_list,
                'target': np.random.choice(target_list)
            })

    with open(item_data_path, 'r', encoding='utf-8') as f:
        item_data = {}
        lines = f.readlines()
        for line in lines:
            #print(line.split('\n')[0])
            item_id, cate_id, pic_url, title = line.split('\n')[0].split(',')[:4]
            if pic_url[:5] != 'http:' and pic_url[:6] != 'https:':
                pic_url = 'http:' + pic_url
            item_data[item_id] = {
                'cate_id': cate_id,
                'pic_url': pic_url,
                'title': title
            }

    return user_data, item_data, image_data_dict


def loadMovieFullData(his_seq_len=HIS_SEQ_LEN['movie']):
    item_data = {}
    keyword_data_path = os.path.join(DATA_PATH_MOVIE, 'keywords_in_id.pkl')
    item_data_path = os.path.join(DATA_PATH_MOVIE, 'info.csv')
    rating_date_path = os.path.join(DATA_PATH_MOVIE, 'ratings.csv')
    image_data_path = os.path.join(DATA_PATH_MOVIE, 'image_data.pkl')
    
    with open(keyword_data_path, 'rb') as f:
        keywords_in_id = pickle.load(f)
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
        
    mv_info = pd.read_csv(item_data_path)
    for index, row in mv_info.iterrows():
        if row['id'] not in keywords_in_id:
            continue
        only_name = re.findall(r'^(.*) \(\d+\) *$', row['name'])
        if len(only_name):
            row['name'] = only_name[0]
        movie_id = str(row['id'])
        item_data[movie_id] = dict(row)
        item_data[movie_id]['keyword'] = keywords_in_id[row['id']]
        

    rating = pd.read_csv(rating_date_path)
    user_mv = {}
    for index, row in rating.iterrows():
        row = dict(row)
        row['movieId'] = str(int(row['movieId']))
        if row['movieId'] in image_data:
            if user_mv.get(row['userId'], None) is None:
                user_mv[row['userId']] = []
            user_mv[row['userId']].append(row)
    
    user_data = []
    for user_id in user_mv:
        mv_list = user_mv[user_id]
        mv_list = [ x for x in mv_list if x['rating']>=4 ]
        if len(mv_list) <= 1:
            continue
        mv_list = sorted(mv_list, key=lambda x: (-x['rating'], -x['timestamp']))
        length = min(his_seq_len, len(mv_list)-1)
        target_mv_id = mv_list[length]['movieId']
        his = [x['movieId'] for x in mv_list[:length]]
        user_data.append({
            'user_id': user_id,
            'history': his,
            'target': target_mv_id
        })
    
    return user_data, item_data, image_data

def loadFullData(dataset='POG', **kw):
    if dataset == 'POG':
        return loadPOGFullData(**kw)
    if dataset == 'movie':
        return loadMovieFullData(**kw)