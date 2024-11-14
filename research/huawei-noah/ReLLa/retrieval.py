import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import json
import mindspore as ms
import mindspore.numpy as msnp


input_dict = {
    "User ID": None,
    "Movie ID": None,
    "Movie title": None,
    "history ID": None,
    "history rating": None,
}

prefer = []
unprefer = []

def get_template(input_dict, temp_type="simple"):

    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item
    low: w/ retrieval, the items is listed with ascending order of similarity to target item
    """

    template = \
{
        "simple": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",


        "sequential": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

        "low": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'][::-1])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

}

    assert temp_type in template.keys(), "Template type error."
    return template[temp_type]



def get_raw_prompt(
    K=15, 
    temp_type="simple", 
):
    global input_dict, template
    df = pd.read_parquet(args.data_dir).reset_index(drop=True)

    id_to_title = json.load(open(args.meta_dir))['id_to_title']

    # fill the template
    for index in tqdm(list(df.index)):
        cur_temp = row_to_prompt(index, df, K, id_to_title, temp_type)
        yield cur_temp



def get_ret_prompt(
    K=15,
    temp_type="sequential"
):
    global input_dict, template
    df = pd.read_parquet(args.data_dir).reset_index(drop=True)
    sorted_indice = json.load(open(args.indice_dir))

    id_to_title = json.load(open(args.meta_dir))['id_to_title']


    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            assert key in row.keys(), "Key name error."
            input_dict[key] = row[key]

        cur_indice = sorted_indice[row_number]
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["history ID"], input_dict["history rating"])}
        if temp_type == "sequential":
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["history ID"])}
        input_dict["history ID"], input_dict["history rating"] = [], []
        
        for i in range(min(K, len(cur_indice))):
           input_dict['history ID'].append(cur_indice[i])
           input_dict['history rating'].append(hist_rating_dict[cur_indice[i]])

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["history ID"], input_dict["history rating"] = map(list, zip(*zipped_list))
        input_dict["history ID"] = list(map(lambda x: id_to_title[str(x)], input_dict["history ID"]))

        for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
            suffix = " stars)" if star > 1 else " star)"
            input_dict["history ID"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)



def row_to_prompt(index, df, K, id_to_title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()
    
    for key in input_dict:
        assert key in row.keys(), "Key name error."
        input_dict[key] = row[key]

    input_dict["history ID"] = list(map(lambda x: id_to_title[str(x)], input_dict["history ID"]))

    input_dict["history ID"] = input_dict["history ID"][-K:]
    input_dict["history rating"] = input_dict["history rating"][-K:]
    for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["history ID"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)



def SUBR(args):
    embeddings = msnp.load(args.embed_dir)
    print(f"Embedding shape: {embeddings.shape}")

    if args.use_pca:
        pca = PCA(n_components=args.pca_dim)
        embeddings = pca.fit_transform(embeddings)
        print("PCA finished.")

    all_indice = []

    df = pd.read_parquet(args.data_dir)

    for _, row in tqdm(df.iterrows()):
        assert "Item ID" in row and "history ID" in row, "Data should have columns 'Item ID' (target Item) and 'history ID' (history items)."
        
        tgt_id = row['Item ID']
        hist_id = row['history ID']
        
        tgt_embed, hist_embed = embeddings[tgt_id], embeddings[hist_id]
        seq_id_to_item_id = {i: item_id for i, item_id in enumerate(hist_id)}
        sim_matrix = msnp.sum(hist_embed * tgt_embed, axis=-1)
        indice = msnp.argsort(-sim_matrix)[:100].asnumpy().tolist()
        sorted_indice = list(map(lambda x: int(seq_id_to_item_id[x]), indice))
        all_indice.append(sorted_indice)  

    json.dump(all_indice, open(args.indice_dir, 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, help="Path to the embeddings. Should be a .npy file.")
    parser.add_argument("--data_dir", type=str, help="Path to the data. Should be a .parquet.gz file.")
    parser.add_argument("--indice_dir", type=str, help="Path to save the SUBR indice.")
    parser.add_argument("--meta_dir", type=str, help="Path to the meta data. Should be a .json file.")
    parser.add_argument("--output_dir", type=str, help="Path to save the prompts.")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=512)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--temp_type", type=str, default="simple", help="Prompt type: simple, sequential, high, low")
    args = parser.parse_args()

    # Retrieval.
    SUBR(args)

    # Generate prompts.
    if args.temp_type == "simple":
        prompt_iter = get_raw_prompt(K=args.K, temp_type=args.temp_type)
    else:
        prompt_iter = get_ret_prompt(K=args.K, temp_type=args.temp_type)


    df = pd.read_parquet(args.data_dir)

    data_list = []
    for msg, idx in zip(prompt_iter, df.index):
        labels = df.loc[idx, "labels"]
        data_dict = {}
        data_dict['input'] = msg
        data_dict['output'] = "Yes." if int(labels) == 1 else "No."
        data_list.append(data_dict)

    assert len(data_list) == len(df.index)

    json.dump(data_list, open(args.output_dir, "w"), indent=4)