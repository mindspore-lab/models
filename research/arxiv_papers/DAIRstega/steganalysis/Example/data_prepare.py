import re
import os
import spacy
import numpy as np

def merge_data(input_files=["test", "train","val"], output_file_path=None,max_triples_num=1,min_triples_num=1):
    merged_sources = []
    merged_targets = []
    for file in input_files:
        if ".debug" not in file:
            source_file = file+".source"
            target_file = file+".target_eval"
        else:
            source_file = file.split("_")[0]+".source"
            target_file = file
        with open(source_file,"r",encoding="utf-8") as f_source, open(target_file, "r", encoding="utf-8") as f_target:
            lines_source = f_source.read().split("\n")
            lines_target = f_target.read().split("\n")

            for source, target in zip(lines_source, lines_target):
                triples = source.split("\n")[0].split("<H>")[1:]
                if len(triples) > max_triples_num or len(triples)<min_triples_num:
                    continue
                else:
                    merged_sources.append(source)
                    merged_targets.append(target)
    if output_file_path is None:
        return merged_sources, merged_targets
    with open(output_file_path+".source","w",encoding="utf-8") as f_source, open(output_file_path+".target","w",encoding="utf-8") as f_target:
        f_source.writelines("\n".join(merged_sources))
        f_target.writelines("\n".join(merged_targets))

def generate_data(neg_file="data/1_source.txt.debug",pos_file="data/mergered_cover_1", output_file="pos.txt"):
    if ".debug" not in neg_file:
        neg_source_file = neg_file + ".source"
        neg_target_file = neg_file + ".target"
    else:
        neg_source_file = neg_file
        neg_target_file = neg_file.replace("source", "target")
    if ".debug" not in pos_file:
        pos_source_file = pos_file+".source"
        pos_target_file = pos_file+".target"
    else:
        pos_source_file = pos_file
        pos_target_file = pos_file.replace("source","target")
    pos_final_sources = []
    pos_final_targets = []
    with open(neg_source_file, "r", encoding="utf-8") as neg_f_source, open(pos_source_file, "r", encoding="utf-8") as pos_f_source, open(pos_target_file,"r",encoding="utf-8") as pos_f_target:
        neg_sources = neg_f_source.read().lower().split("\n")
        pos_sources = pos_f_source.read().lower().split("\n")
        pos_targets = pos_f_target.read().lower().split("\n")
        for pos_source, pos_target in zip(pos_sources,pos_targets):
            if pos_source in neg_sources:
                pos_final_sources.append(pos_source)
                pos_final_targets.append(pos_target)
            else:
                continue
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pos_final_targets))

def textParse(str_doc):
    # 正则过滤掉特殊符号、标点、英文、数字等。
    r1 = '[’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # 去除换行符
    str_doc=re.sub(r1, '', str_doc)
    # 多个空格成1个
    words = str_doc.split(" ")
    new_words = []
    for word in words:
        if word =="":
            continue
        else:
            new_words.append(word)
    str_doc = " ".join(new_words)
    return str_doc

def filter_text_by_cat(reference_source_file = "data/category/Airport.source", text_source_file= "data/mergered_cover_1_to_1.source", text_file="data/mergered_cover_1_to_1.target", label="neg"):
    with open(reference_source_file, "r", encoding="utf-8") as f_reference_source, open(text_source_file, "r", encoding="utf-8") as f_text_source, open(text_file, "r",encoding="utf-8") as f_text:
        reference_source = f_reference_source.read().split("\n")
        texts = f_text.read().split("\n")
        texts_source = f_text_source.read().split("\n")

    texts_reference_source=[]
    filt_reference_source=[]
    # for i, reference_source_ in enumerate(reference_source):
    #     reference_source[i] = textParse(reference_source_)
    for text, text_source in zip(texts, texts_source):
        # if textParse(text_source) in reference_source:
        #     texts_reference_source.append(text)
        if text_source in reference_source:
            texts_reference_source.append(text)
            filt_reference_source.append(text_source)

    text_reference_source_file = os.path.join(os.path.split(reference_source_file)[0], label+"_"+os.path.basename(reference_source_file).split(".")[0]+".target")
    filt_reference_source_file = os.path.join(os.path.split(reference_source_file)[0],
                                              "filter_"+label + "_" + os.path.basename(reference_source_file))
    with open(text_reference_source_file, "w", encoding="utf-8") as f, open(filt_reference_source_file, "w", encoding="utf-8") as f_filt:
        f.write("\n".join(texts_reference_source))
        f_filt.write("\n".join(filt_reference_source))

def extract_eitities(sentences, model="en_core_web_lg"):
    nlp = spacy.load("en_core_web_lg")
    static_info = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        static_info.append(len(entities))
    static_info = np.array(static_info)
    return (static_info, static_info.mean(), static_info.std(),np.argwhere(static_info!=0).size)

def convert_source_target_to_dict(source_file= "data/mergered_cover_1_to_1.source", target_file="data/mergered_cover_1_to_1.target"):
    data = dict()
    with open(source_file, "r", encoding="utf-8") as f_source, open(target_file, "r",encoding="utf-8") as f_target:
        sources = f_source.read().split("\n")
        targets = f_target.read().split("\n")
    for source, target in zip(sources, targets):
        if data.get(source, None) is None:
            data[source] = [target]
        else:
            data[source].append(target)
    return data

def sents_match(output_source_file="data/category/Airport.source", output_target_file="data/category/neg_Airport.target",
                ref_source_file= "data/mergered_cover_1_to_1.source", ref_target_file="data/mergered_cover_1_to_1.target",
                output_sents_file="data/bleu/Airport.output", ref_sents_file="data/bleu/Airport.ref"):
    output_data = convert_source_target_to_dict(output_source_file, output_target_file)
    ref_data = convert_source_target_to_dict(ref_source_file, ref_target_file)
    with open(output_sents_file,"w",encoding="utf-8") as f_sents, open(ref_sents_file, "w",encoding="utf-8") as f_ref:
        for triple, sent in output_data.items():
            f_sents.write("\t".join(sent)+"\n")
            if ref_data.get(triple, None) is None:
                f_ref.write("\n")
            else:
                refs = ref_data[triple]
                f_ref.write("\t".join(refs)+"\n")
    return



if __name__ == '__main__':
    # for max_num in range(1,6):
    #     for min_num in range(1,max_num+1):
    #         output_file_path = "data/mergered_cover_"+str(min_num)+"_to_"+str(max_num)
    #         merge_data(input_files=["data/train","data/val","data/test_both"],output_file_path=output_file_path,
    #                    min_triples_num=min_num,max_triples_num=max_num)
    # neg_file_paths = ["data/1_source.txt.debug","data/2_source.txt.debug","data/3_source.txt.debug","data/4_source.txt.debug","data/5_source.txt.debug"]
    # for max_num in range(1,6):
    #     for min_num in range(1,max_num+1):
    #         output_file_path = "data/Graph_stego_"+str(min_num)+"_to_"+str(max_num)
    #         merge_data(input_files=neg_file_paths, output_file_path=output_file_path, max_triples_num=max_num, min_triples_num=min_num)
    # generate_data(neg_file="data/1",pos_file="data/mergered_cover_1", output_file="data/pos_test.txt")

    #
    # dataset = "./data/category"
    # files = [os.path.join(dataset,x) for x in os.listdir(dataset) if os.path.isfile(os.path.join(dataset,x)) and ".source" in os.path.basename(x) and "filter" not in os.path.basename(x)]
    # for file in files:
    #     neg_file = os.path.join(os.path.split(file)[0], "neg_"+os.path.basename(file).split(".")[0]+".target")
    #     pos_file = os.path.join(os.path.split(file)[0], "pos_"+os.path.basename(file).split(".")[0]+".target")
    #     print("python runc.py --neg_filename %s --pos_filename %s"%(neg_file, pos_file))
    #     print("python runc.py --neg_filename %s --pos_filename ./data/mergered_cover_1_to_1.target"%neg_file)
    #     filter_text_by_cat(reference_source_file=file, label="pos")
    #     filter_text_by_cat(reference_source_file=file, label="neg", text_file="data/Graph_stego_1_to_1.target", text_source_file="data/Graph_stego_1_to_1.source")


    # with open("./data/Graph_stego_1_to_1.target", "r", encoding="utf-8") as f:
    #     sentences = f.read().split("\n")
    #     static_info = extract_eitities(sentences)
    # print(static_info)
    # print("prepare data test")

    dataset = "./data/category"
    dataset_bleu = "./data/bleu"
    files = [os.path.join(dataset,x) for x in os.listdir(dataset) if os.path.isfile(os.path.join(dataset,x)) and ".source" in os.path.basename(x)  and "filter" not in x]
    for file in files:
        source_file = os.path.join(os.path.split(file)[0], "filter_neg_"+os.path.basename(file).split(".")[0]+".source")
        target_file = os.path.join(os.path.split(file)[0], "neg_"+os.path.basename(file).split(".")[0]+".target")
        output_sents_file = os.path.join(dataset_bleu, os.path.basename(file).split(".source")[0]+".output")
        refs_sents_file = os.path.join(dataset_bleu, os.path.basename(file).split(".source")[0]+".refs")
        sents_match(output_source_file=source_file,output_target_file=target_file,output_sents_file=output_sents_file,ref_sents_file=refs_sents_file)
    source_file = "data/Graph_stego_1_to_1.source"
    target_file = "data/Graph_stego_1_to_1.target"
    output_sents_file = "data/bleu/Graph_stego_1_to_1.output"
    refs_sents_file = "data/bleu/Graph_stego_1_to_1.refs"
    sents_match(output_source_file=source_file, output_target_file=target_file, output_sents_file=output_sents_file,
                ref_sents_file=refs_sents_file)