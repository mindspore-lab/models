import math
import torch
import mauve
import numpy as np
from peft import PeftModel
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertModel, BertTokenizer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from steganography.ADG import lm, utils
from itertools import islice
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.matutils import kullback_leibler, sparse2full
from scipy.spatial.distance import cosine


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


# ========================== main ==========================
def main(name, model):
# def main(name):

    cover_text_path = './data_cover/A_Overall'
    stego_text_path = './data_stego/adg065/' + name + "/" + name                # Ours
    # stego_text_path = './data_stego/topk/a48-sqrt/' + name + "/" + name                # Ours
    # stego_text_path = './data_stego/ADG/adg'                                   # ADG
    # stego_text_path = './data_stego/Tina-fang/Separate/A_Overall/A_Overall_1'  # Tina-Fang
    # stego_text_path = './data_stego/RNN-Stega/Separate/bit_5/A_Overall/rnn'  # RNN-Stega
    # stego_text_path = './data_stego/PLMmark/stego'
    # stego_text_path = './data_stego/LLsM/48'

    # ----- text loading -----
    with open(stego_text_path + ".txt", encoding='utf-8') as file:
        tex_stego = file.read()
    with open(cover_text_path + ".txt", encoding='utf-8') as file:
        tex_cover = file.read()
    
    # tex_stego = "Shakespeare's writing style is unique and unmatched by any other writer in history. His plays are notable for their richness and variety, their use of vivid imagery and metaphor, their powerful characters and their insight into the human condition. Shakespeare's use of iambic pentameter, for example, adds to the rhythm and beauty of his verse, while his use of puns and wordplay adds to the humor and wit of his plays. His characters are complex and multifaceted, and his use of language to explore their emotions and motivations is unparalleled. Shakespeare's writing style is also notable for its use of language to explore themes of love, loss, betrayal, and redemption. His plays are filled with powerful imagery and metaphor that help to bring these themes to life. Shakespeare's writing style is also notable for its use of language to explore the human condition. His characters are complex and multifaceted, and his use of language to explore their emotions and motivations is unparalleled."

    # PPL_cover = []
    # with open(cover_text_path + ".txt", encoding='utf-8') as file:
    #     while True:
    #         tex_c = list(islice(file, 5))
    #         if not tex_c:
    #             break
    #         tex_co = ''.join(tex_c)
    #         PPL_cover.append(ppl(tex_co, model))
    #     PPL_cove = sum(PPL_cover) / len(PPL_cover)
    # print(PPL_cove)

    PPL_stego = []
    with open(stego_text_path + ".txt", encoding='utf-8') as file:
        while True:
            tex_c = list(islice(file, 5))
            if not tex_c:
                break
            tex_co = ''.join(tex_c)
            PPL_stego.append(ppl(tex_co, model))
        PPL_steg = sum(PPL_stego) / len(PPL_stego)
    print(PPL_steg)

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
    # delta_MP = np.mean(PPL_steg) - np.mean(PPL_cove)
    delta_MP = np.mean(PPL_steg) - 7.6924917034
    print(delta_MP, "dPPL")

    # ----- 4. KLD (Input: texts_cover: list-list-str || texts_stego: list-list-str || prob_cover: ndarray || prob_stego: ndarray) -----
    # ----- 5. JSD (Input: texts_cover: list-list-str || texts_stego: list-list-str || prob_cover: ndarray || prob_stego: ndarray) -----
    frequency_stego = create_frequency_distribution(texts_stego)
    frequency_cover = create_frequency_distribution(texts_cover)
    probability_stego = normalize_distribution(frequency_stego)
    probability_cover = normalize_distribution(frequency_cover)
    vocab = set(probability_stego.keys()).union(set(probability_cover.keys()))
    epsilon = 1e-10
    prob_stego = np.array([probability_stego.get(word, epsilon) for word in vocab])
    prob_cover = np.array([probability_cover.get(word, epsilon) for word in vocab])
    KLD = kl_divergence(prob_cover, prob_stego)
    JSD = np.sqrt(js_divergence(prob_stego, prob_cover))

    # 余弦相似度
    def join_words(word_list):
        return ' '.join(word_list)

    combined_text1 = ' '.join([join_words(text) for text in texts_cover])
    combined_text2 = ' '.join([join_words(text) for text in texts_stego])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([combined_text1, combined_text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print("Δ:", 1 - cosine_sim[0][0])
    vector1 = tfidf_matrix[0].toarray()[0]
    vector2 = tfidf_matrix[1].toarray()[0]
    euclidean_dist = euclidean(vector1, vector2)
    print("", euclidean_dist)

    manhattan_dist = cityblock(vector1, vector2)
    print("", manhattan_dist)

    dot_product = np.dot(vector1, vector2)
    print("Δ：", 1 - dot_product)

    # LDA
    def preprocess_tokenized(text_list):
        return ' '.join(text_list)
    processed_text1 = [preprocess_tokenized(text) for text in texts_cover]
    processed_text2 = [preprocess_tokenized(text) for text in texts_stego]
    # processed_text1 = preprocess(texts_cover)
    # processed_text2 = preprocess(texts_stego)
    dictionary = corpora.Dictionary([processed_text1, processed_text2])
    corpus = [dictionary.doc2bow(text) for text in [processed_text1, processed_text2]]
    lda = LdaModel(corpus, num_topics=70, id2word=dictionary)
    topic_distribution1 = lda.get_document_topics(corpus[0], minimum_probability=0)
    topic_distribution2 = lda.get_document_topics(corpus[1], minimum_probability=0)
    num_topics = lda.num_topics
    vec1 = sparse2full(topic_distribution1, num_topics)
    vec2 = sparse2full(topic_distribution2, num_topics)
    similarity = 1 - cosine(vec1, vec2)
    print(f"LDA: {similarity}")


    # ----- 6. BLEU (Input: texts_cover: list-list-str || texts_stego: list-list-str || candidate: list-str) -----
    # reference = [tex_cover]
    # candidate = tex_stego
    # bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    #
    # # ----- 7. BERTScore (Input: tex_cover: str || tex_stego: str) -----
    # BERTScore1, BERTScore2, BERTScore3 = scorer.score([tex_stego], [tex_cover])
    # BERTScore3 = BERTScore3.item()
    #
    # # ----- 8. Rouge-L (Input: texts_cover: list-list-str || texts_stego: list-list-str) -----
    # rouge_l_scores = []
    # for mt, rt in zip(texts_stego, texts_cover):
    #     score = rouge_l_score(mt, rt)
    #     rouge_l_scores.append(score)
    # average_precision = sum(score[0] for score in rouge_l_scores) / len(rouge_l_scores)
    # average_recall = sum(score[1] for score in rouge_l_scores) / len(rouge_l_scores)
    # Rouge_L = sum(score[2] for score in rouge_l_scores) / len(rouge_l_scores)
    #
    # # ----- 10. MAUVE (Input: p_text: list-str || q_text: ist-str) -----
    # with open(stego_text_path + ".txt", encoding='utf-8') as file:
    #     tex_stego = file.readlines()
    # with open(cover_text_path + ".txt", encoding='utf-8') as file:
    #     tex_cover = file.readlines()
    # Mauve = mauve.compute_mauve(p_text=tex_cover, q_text=tex_stego, device_id=0, batch_size=8)  # p_text: 人类 q_text: 机器
    #
    # # ----- 9. CIDEr (Input: texts_stego: list-list-str || candidate: list-str) -----
    # # CIDEr = calculate_cider_like_score(texts_stego, candidate)
    #
    # # ----- 11. METEOR (Input: texts_cover: list-list-str || texts_stego: list-list-str) -----
    # meteor_scores = []
    # for mt, rt in zip(texts_stego, texts_cover):
    #     score = meteor_score([rt], mt)
    #     meteor_scores.append(score)
    # METEOR = sum(meteor_scores)/len(meteor_scores)

    # print
    print(name)
    print("2. PPL↓:", PPL_stego,
          # "|| 3. ΔP↓:", delta_MP, "|| 4. KLD↓:", KLD, "|| 5. JSD↓:", JSD
          )
    print("|| 4. KLD↓:", KLD, "|| 5. JSD↓:", JSD)
    # print("6. BLEU↑:", bleu_score, "|| 7. BERTScore↑:", BERTScore1, BERTScore2, BERTScore3)
    # print("6. BLEU↑:", bleu_score, "|| 7. BERTScore↑:", BERTScore1, BERTScore2, BERTScore3, "|| 8. Rouge-L↑:", Rouge_L)

    # print("6. BLEU↑: ", round(bleu_score*100, 2), " || 7. BERTScore↑: ", round(BERTScore3*100, 2),
    #       " || 8. Rouge-L↑: ", round(Rouge_L*100, 2),
    #       " || 10. MAUVE↑: ", round(Mauve.mauve*100, 2), " || 11. METEOR↑: ", round(METEOR*100, 2))


if __name__ == "__main__":
    names = [
             'A_Overall'
    ]
    # ----- parameter setting -----
    load_8bit: bool = True
    base_model: str = "../../LLM/LLaMA2-7B"
    lora_weights: str = "./ft-model"

    # ----- model loading -----
    model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto")
    # model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # scorer = BERTScorer(model_type="bert-base-uncased", lang='en')

    for name in names:
        main(name=name, model=model)
        # main(name=name)
