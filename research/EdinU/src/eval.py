# eval_ms.py
# run intrinisic evaluation, lexical + sts (where available)
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

from utils import create_sts_dataloader
from models import Banyan
from scipy import stats
from bpemb import BPEmb
import numpy as np
import sys
import argparse

class IntrinsicEvaluator:
    def __init__(self, lang):
        self.lang = lang

        if self.lang == 'en':
            # lexical 
            self.sl_data = self.load_word_level('../data/simlex.tsv')
            self.ws_data = self.load_word_level('../data/wordsim_similarity_goldstandard.txt')
            self.wr_data = self.load_word_level('../data/wordsim_relatedness_goldstandard.txt')
            # sentence
            self.sts12_dataloader = create_sts_dataloader('../data/sts12_test.csv', 128)
            self.sts13_dataloader = create_sts_dataloader('../data/sts13_test.csv', 128)
            self.sts14_dataloader = create_sts_dataloader('../data/sts14_test.csv', 128)
            self.sts15_dataloader = create_sts_dataloader('../data/sts15_test.csv', 128)
            self.sts16_dataloader = create_sts_dataloader('../data/sts16_test.csv', 128)
            self.stsb_dataloader = create_sts_dataloader('../data/stsb_test.csv', 128)
            self.sick_dataloader = create_sts_dataloader('../data/sick_test.csv', 128)
            self.sem_dataloader = create_sts_dataloader('../data/semrel_test.csv', 128)
        
        # Cross-lingual STS datasets
        elif self.lang in ['af', 'id', 'te', 'ar', 'hi', 'mr', 'ha', 'am', 'es']:
            self.lang_dataloader = create_sts_dataloader(f'../data/{self.lang}_test.csv', 128, lang=self.lang)
        
    def load_word_level(self, path):
        dataset = []
        bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
        with open(path, 'r') as f:
            for line in f.readlines():
                # dataset.append((bpemb_en.encode_ids(line.split()[0].lower()), bpemb_en.encode_ids(line.split()[1].lower()), line.split()[2]))
                # [FIX] Ensure the score is converted to float
                dataset.append((bpemb_en.encode_ids(line.split()[0].lower()), bpemb_en.encode_ids(line.split()[1].lower()), float(line.split()[2])))
        return dataset

    def embed_word(self, word, model, embeddings):
        model.set_train(False)
        if len(word) > 1:
            word_tensor = Tensor(np.array(word), dtype=ms.int32)
            embed = model(word_tensor, words=True)
        else:
            embed = embeddings[word[0]]
        # [FIX] Removed .cpu()
        return embed

    def evaluate_word_level(self, model):
        # [FIX] model.embedding.weight -> model.embedding.embedding_table, removed .detach().cpu()
        embeddings = model.embedding.embedding_table
        
        # [FIX] Replaced self.cos_words with the correct functional operator ops.cosine_similarity.
        sl_predictions = [ops.cosine_similarity(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model, embeddings), dim=0).asnumpy().item()
                          for x in self.sl_data]
        sl_score = stats.spearmanr(np.array(sl_predictions), np.array([x[2] for x in self.sl_data]))
        print('SimLex Score: {}'.format(sl_score.correlation.round(3)), flush=True)

        ws_predictions = [ops.cosine_similarity(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model, embeddings), dim=0).asnumpy().item()
                          for x in self.ws_data]
        ws_score = stats.spearmanr(np.array(ws_predictions), np.array([x[2] for x in self.ws_data]))
        print('Wordsim S Score: {}'.format(ws_score.correlation.round(3)), flush=True)

        wr_predictions = [ops.cosine_similarity(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model, embeddings), dim=0).asnumpy().item()
                          for x in self.wr_data]
        wr_score = stats.spearmanr(np.array(wr_predictions), np.array([x[2] for x in self.wr_data]))
        print('Wordsim R Score: {}'.format(wr_score.correlation.round(3)), flush=True)

        return sl_score.correlation, ws_score.correlation, wr_score.correlation

    def embed_sts(self, model, dataloader):
        model.set_train(False)
        predicted_sims = []
        all_scores = []
        for tokens_1, tokens_2, scores in dataloader.create_tuple_iterator():
            # [FIX] Convert numpy arrays from dataloader to Tensors, remove .to(device)
            out = model(Tensor(tokens_1), seqs2=Tensor(tokens_2))
            # [FIX] Replaced self.cos_sents with the correct functional operator ops.cosine_similarity.
            sim = ops.cosine_similarity(out[0], out[1], dim=1)
            predicted_sims.append(sim)
            all_scores.append(scores) # scores is already a numpy array
        return predicted_sims, all_scores

    def evaluate_sts(self, model):
        # [FIX] Removed device parameter
        dataloaders = {
            "STS-12": self.sts12_dataloader, "STS-13": self.sts13_dataloader,
            "STS-14": self.sts14_dataloader, "STS-15": self.sts15_dataloader,
            "STS-16": self.sts16_dataloader, 
            "STS-B": self.stsb_dataloader,
            "SICK-R": self.sick_dataloader, "SemRel": self.sem_dataloader
        }
        all_scores_corr = []
        for name, loader in dataloaders.items():
            predicted_sims, ground_truth_scores = self.embed_sts(model, loader)
            # [FIX] torch.cat -> ops.cat, .cpu() -> .asnumpy(), dim -> axis
            # Use np.concatenate for list of numpy arrays
            predicted_np = ops.cat(predicted_sims, axis=0).asnumpy()
            ground_truth_np = np.concatenate(ground_truth_scores, axis=0)
            
            score = stats.spearmanr(predicted_np, ground_truth_np)
            print('{}: {}'.format(name, score.correlation.round(3)), flush=True)
            all_scores_corr.append(score.correlation)
        
        return tuple(all_scores_corr)

    def evaluate_lang(self, model):
        predicted_sims, all_scores = self.embed_sts(model, self.lang_dataloader)
        # [FIX] Corrected conversion from MindSpore/NumPy to SciPy format
        predicted_np = ops.cat(predicted_sims, axis=0).asnumpy()
        ground_truth_np = np.concatenate(all_scores, axis=0)
        
        str_score = stats.spearmanr(predicted_np, ground_truth_np)
        print('SCORE: {}'.format(str_score.correlation.round(3)), flush=True)
        return str_score.correlation

def main():
    # [NEW] Added argparse for runnable script
    parser = argparse.ArgumentParser(description="MindSpore Banyan Model Intrinsic Evaluation")
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file (.pth)')
    parser.add_argument('--lang', type=str, default='en', help='Language for evaluation (e.g., en, af, id)')
    parser.add_argument('--e_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--channels', type=int, default=128, help='Model channels')
    parser.add_argument('--r', type=float, default=0.1, help='R parameter for the model')
    args = parser.parse_args()

    # [FIX] Set MindSpore context. No device parameters needed.
    ms.set_context(mode=ms.GRAPH_MODE)

    # [FIX] Model instantiation without device assignment
    model = Banyan(vocab_size=25001, embedding_size=args.e_dim, channels=args.channels, r=args.r)
    
    # Load weights from the provided checkpoint path
    print("Loading weights from checkpoint...")
    param_dict = ms.load_checkpoint(args.checkpoint_path)
    ms.load_param_into_net(model, param_dict)
    print("Weight loading complete.")

    # [FIX] Evaluator instantiation without device assignment
    evaluator = IntrinsicEvaluator(lang=args.lang)

    # [FIX] Removed training loop. This is an evaluation script.
    print("\n--- Starting Evaluation ---", flush=True)
    if args.lang == 'en':
        
        print('Lexical Evaluation', flush=True)
        sl_score, ws_score, wr_score = evaluator.evaluate_word_level(model)
        lex_score = (sl_score + ws_score + wr_score) / 3
        print('Average Lex Score: {:.3f}'.format(lex_score), flush=True)
        print('\n')
        '''
        '''
        print('STS Evaluation', flush=True) 
        scores = evaluator.evaluate_sts(model)
        sts_score = np.mean(scores)
        print('Average STS Score: {:.3f}'.format(sts_score), flush=True)
        print('\n')
        
    
    else:
        print(f'{args.lang.upper()} STR Evaluation', flush=True)
        str_score = evaluator.evaluate_lang(model)
        print(f'Final Score for {args.lang.upper()}: {str_score:.3f}')
    
    print("--- Evaluation Complete ---", flush=True)

if __name__ == "__main__":
    main()
        
