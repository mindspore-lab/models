import math
import torch
import mauve
import csv
import numpy as np
from peft import PeftModel
from bert_score import BERTScorer
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BertModel, BertTokenizer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from steganography.ADG import lm, utils
from itertools import islice

# ------------ PPL ------------
def ppl(text, model):
    inputs = tokenizer(text, return_tensors="pt")
    # inputs = torch.LongTensor([[vocabulary.w2i[text]]])
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity


# ------------ ER ------------
# def er(texts, bits):
#     result = {}
#     em = []
#     for i in range(len(texts)):
#         em.append(len(bits[i]) / (len(texts[i]) - 1))  # drop start word
#     result['.'.join(str(file).split('.')[:-1])] = {}
#     result['.'.join(str(file).split('.')[:-1])]['mean'] = np.mean(em)
#     result['.'.join(str(file).split('.')[:-1])]['std'] = np.std(em, ddof=1)
#     return result['.'.join(str(file).split('.')[:-1])]['mean']


# ------------ KLD ------------
def create_frequency_distribution(tokenized_texts):
    frequency_distribution = Counter()
    for sentence in tokenized_texts:
        frequency_distribution.update(sentence)
    return frequency_distribution

def normalize_distribution(frequency_distribution):
    total_count = sum(frequency_distribution.values())
    return {word: count / total_count for word, count in frequency_distribution.items()}

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# ------------ JSD ------------
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


# ------------ Rouge-L ------------
def rouge_l_score(candidates, reference):
    def lcs(x, y):
        m, n = len(x), len(y)
        c = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    c[i][j] = c[i - 1][j - 1] + 1
                else:
                    c[i][j] = max(c[i - 1][j], c[i][j - 1])
        return c[m][n]
    lcs_length = lcs(candidates, reference)
    precision = lcs_length / len(candidates)
    recall = lcs_length / len(reference)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score


#  ------------ CIDEr  ------------
def calculate_cider_like_score(candidates, reference, n=4):
    candidates = [cand.lower() for cand in candidates if isinstance(cand, str)]
    reference = [ref.lower() for ref in reference]
    combined_sentences = candidates + reference
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, n))
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_sentences)
    cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    # for i in range(len(candidates)):
    #     cosine_score = F.cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[len(candidates):])
    #     cosine_scores.append(np.mean(cosine_score))
    return np.mean(cosine_scores)


# ========================== main ==========================
def main(name, scorer):
    cover_text_path = './data_cover/' + name
    stego_text_path = './data_stego/adg065/' + name + "/" + name  # Ours

    # ----- text loading -----
    with open(stego_text_path + ".txt", encoding='utf-8') as file:
        tex_stego = file.read()
    with open(cover_text_path + ".txt", encoding='utf-8') as file:
        tex_cover = file.read()

    # tex_stego = "Shakespeare's writing style is a rich tapestry of imagery, metaphor, and wordplay. His characters are often larger than life, with a complexity and depth that is rarely seen in literature. He explores themes such as love, loss, ambition, revenge, and mortality with a poetic flair that is unmatched. Shakespeare's use of language is particularly noteworthy. He employs a wide range of literary devices, including iambic pentameter, rhyme, and allusion, to create a sense of musicality and depth in his works. Shakespeare's plays are often characterized by their dramatic structure, with complex plots, multiple characters, and a range of emotions that are expertly woven together. His characters are often deeply flawed, yet their struggles and triumphs are relatable and resonant."

    # PPL_cover = []
    # with open(cover_text_path + ".txt", encoding='utf-8') as file:
    #     while True:
    #         tex_c = list(islice(file, 10))
    #         if not tex_c:
    #             break
    #         tex_co = ''.join(tex_c)
    #         PPL_cover.append(ppl(tex_co, model))
    #     PPL_cove = sum(PPL_cover) / len(PPL_cover)
    # print(PPL_cove)
    #
    # PPL_stego = []
    # with open(stego_text_path + ".txt", encoding='utf-8') as file:
    #     while True:
    #         tex_c = list(islice(file, 10))
    #         if not tex_c:
    #             break
    #         tex_co = ''.join(tex_c)
    #         PPL_stego.append(ppl(tex_co, model))
    #     PPL_steg = sum(PPL_stego) / len(PPL_stego)
    # print(PPL_steg)

    with open(stego_text_path + ".txt", encoding='utf-8') as files:
        texts_stego = files.readlines()
    texts_stego = [_.strip() for _ in texts_stego]
    texts_stego = list(map(lambda x: x.split(), texts_stego))
    with open(cover_text_path + ".txt", encoding='utf-8') as files:
        texts_cover = files.readlines()
    texts_cover = [_.strip() for _ in texts_cover]
    texts_cover = list(map(lambda x: x.split(), texts_cover))
    # with open(stego_text_path + ".bit", encoding='utf8') as f:
    #     bits = f.readlines()
    # bits = [_.strip() for _ in bits]

    # # ----- 1. bpw (Input: texts_stego: list-list-str || bits: list-str) -----
    # bpw = er(texts_stego, bits)
    #
    # # ----- 2. PPL (Input: tex_stego: str || model: PeftModelForCausalLM) -----
    # PPL_stego = ppl(tex_stego, model)
    #
    # # ----- 3. ΔP (Input: tex_cover: str || tex_stego: str || model: PeftModelForCausalLM) -----
    # PPL_cover = ppl(tex_cover, model)
    # delta_MP = np.mean(PPL_stego) - np.mean(PPL_cover)

    # ----- 4. KLD (Input: texts_cover: list-list-str || texts_stego: list-list-str || prob_cover: ndarray || prob_stego: ndarray) -----
    # ----- 5. JSD (Input: texts_cover: list-list-str || texts_stego: list-list-str || prob_cover: ndarray || prob_stego: ndarray) -----
    # frequency_stego = create_frequency_distribution(texts_stego)
    # frequency_cover = create_frequency_distribution(texts_cover)
    # probability_stego = normalize_distribution(frequency_stego)
    # probability_cover = normalize_distribution(frequency_cover)
    # vocab = set(probability_stego.keys()).union(set(probability_cover.keys()))
    # epsilon = 1e-10
    # prob_stego = np.array([probability_stego.get(word, epsilon) for word in vocab])
    # prob_cover = np.array([probability_cover.get(word, epsilon) for word in vocab])
    # KLD = kl_divergence(prob_stego, prob_cover)
    # JSD = np.sqrt(js_divergence(prob_stego, prob_cover))

    # ----- 6. BLEU (Input: texts_cover: list-list-str || texts_stego: list-list-str || candidate: list-str) -----
    reference = [tex_cover] 
    candidate = tex_stego 
    bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

    # ----- 7. BERTScore (Input: tex_cover: str || tex_stego: str) -----
    BERTScore1, BERTScore2, BERTScore3 = scorer.score([tex_stego], [tex_cover])
    BERTScore3 = BERTScore3.item()

    # ----- 8. Rouge-L (Input: texts_cover: list-list-str || texts_stego: list-list-str) -----
    rouge_l_scores = []
    for mt, rt in zip(texts_stego, texts_cover):
        score = rouge_l_score(mt, rt)
        rouge_l_scores.append(score)
    average_precision = sum(score[0] for score in rouge_l_scores) / len(rouge_l_scores)
    average_recall = sum(score[1] for score in rouge_l_scores) / len(rouge_l_scores)
    Rouge_L = sum(score[2] for score in rouge_l_scores) / len(rouge_l_scores)

    # ----- 10. MAUVE (Input: p_text: list-str || q_text: ist-str) -----
    with open(stego_text_path + ".txt", encoding='utf-8') as file:
        tex_stego = file.readlines()
    with open(cover_text_path + ".txt", encoding='utf-8') as file:
        tex_cover = file.readlines()
    Mauve = mauve.compute_mauve(p_text=tex_cover, q_text=tex_stego, device_id=0, batch_size=4)  # p_text: 人类 q_text: 机器

    # ----- 9. CIDEr (Input: texts_stego: list-list-str || candidate: list-str) -----
    # CIDEr = calculate_cider_like_score(texts_stego, candidate)

    # ----- 11. METEOR (Input: texts_cover: list-list-str || texts_stego: list-list-str) -----
    meteor_scores = []
    for mt, rt in zip(texts_stego, texts_cover):
        score = meteor_score([rt], mt)
        meteor_scores.append(score)
    METEOR = sum(meteor_scores)/len(meteor_scores)

    # print
    print(name)
    # print("2. PPL↓:", PPL_stego, "|| 3. ΔP↓:", delta_MP, "|| 4. KLD↓:", KLD, "|| 5. JSD↓:", JSD)
    # print("6. BLEU↑:", bleu_score, "|| 7. BERTScore↑:", BERTScore1, BERTScore2, BERTScore3)
    # print("6. BLEU↑:", bleu_score, "|| 7. BERTScore↑:", BERTScore1, BERTScore2, BERTScore3, "|| 8. Rouge-L↑:", Rouge_L)
    print("6. BLEU↑: ", round(bleu_score*100, 2), " || 7. BERTScore↑: ", round(BERTScore3*100, 2),
          " || 8. Rouge-L↑: ", round(Rouge_L*100, 2),
          " || 10. MAUVE↑: ", round(Mauve.mauve*100, 2), " || 11. METEOR↑: ", round(METEOR*100, 2))

    return round(bleu_score*100, 2), round(BERTScore3*100, 2), round(Rouge_L*100, 2), round(Mauve.mauve*100, 2), round(METEOR*100, 2)


if __name__ == "__main__":
    discourse = []
    names = [
             'G_movie', 'G_news', 'G_twitter',  'P_Andersen',  'P_Conan',  'P_Dickens',  'P_Shakespeare',  'P_Tolstoy',
             'T_administration', 'T_animals', 'T_anime', 'T_architecture', 'T_art', 'T_astronomy', 'T_automotive',
             'T_beauty', 'T_biology', 'T_chemistry', 'T_civil', 'T_computer_science', 'T_consumption', 'T_culture',
             'T_current_affairs', 'T_dance', 'T_design', 'T_drinks', 'T_earth_sciences', 'T_economy', 'T_education',
             'T_emotion', 'T_engineering', 'T_entertainment', 'T_family', 'T_finance', 'T_food', 'T_game',
             'T_geography', 'T_gourmet', 'T_health', 'T_history', 'T_humanities', 'T_interpersonal', 'T_judicial',
             'T_language', 'T_law', 'T_life', 'T_literature', 'T_love', 'T_mathematics', 'T_medical', 'T_military',
             'T_movies', 'T_music', 'T_nature', 'T_pets', 'T_philosophy', 'T_physics', 'T_plants', 'T_politics',
             'T_psychology', 'T_real_estate', 'T_regulations', 'T_religion', 'T_shopping', 'T_society', 'T_sport',
             'T_technology', 'T_TV', 'T_war', 'T_work'
             ]
    # ----- parameter setting -----
    # load_8bit: bool = True
    base_model: str = "../../LLM/LLaMA2-7B"
    # lora_weights: str = "./model_output/steganography_finetune/13B-styleme-8-qv"

    # ----- model loading -----
    # model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto")
    # model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    scorer = BERTScorer(model_type="bert-base-uncased", lang='en')

    for name in names:
        a = main(name=name, scorer=scorer)
        discourse.append(a)
    print(discourse)

    filename = "output51.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in discourse:
            writer.writerow(row)
    print(f"{filename}")
