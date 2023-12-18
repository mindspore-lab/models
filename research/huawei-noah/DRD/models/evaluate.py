import mindspore
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn import metrics
from sklearn.utils import shuffle
import sys
sys.path.append("..")
from utils.metircs import NDCG,ARP,AP,MRR,Precision

class Evaluator(object):
    
    def __init__(self, eval_log, eval_positions, use_cuda = False):
        self.use_cuda = use_cuda
        self.eval_log = eval_log
        self.eval_positions = eval_positions
        
        self.NDCG_evals = []
        #self.ARP_evals = []
        self.MRR_evals = []
        #self.MAP_evals = []
        self.Precision_evals = []
        for p in eval_positions:
            self.NDCG_evals.append(NDCG(p))
            #self.ARP_evals.append(ARP(p))
            self.MRR_evals.append(MRR(p))
            #self.MAP_evals.append(AP(p))
            self.Precision_evals.append(Precision(p))
    
    def evaluate(self, model):
        eval_predicts = self._get_predicts(model)
        df_eval, df_stat = self._get_stat(eval_predicts)

        #AUC = metrics.roc_auc_score(df_eval['label'], df_eval['predict'])
        NDCGs = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
                 for NDCG_eval in self.NDCG_evals]
        # ARPs = [df_stat['label_list'].apply(ARP_eval.evaluate).mean() 
        #              for ARP_eval in self.ARP_evals]
        MRRs = [df_stat['label_list'].apply(MRR_eval.evaluate).mean() 
                for MRR_eval in self.MRR_evals]
        # MAPs = [df_stat['label_list'].apply(MAP_eval.evaluate).mean() 
        #              for MAP_eval in self.MAP_evals]
        Precisions = [df_stat['label_list'].apply(Precision_eval.evaluate).mean() 
                        for Precision_eval in self.Precision_evals]

        # wrap eval result into pandas dataframe
        eval_result = pd.DataFrame([],columns=['NDCG','MRR','Precision'], index= self.eval_positions)
        eval_result['NDCG'] = NDCGs
        eval_result['MRR'] = MRRs
        eval_result['Precision'] = Precisions

        return eval_result
        
        
    def _get_predicts(self, model):
        # with torch.no_grad():
        #     if self.use_cuda:
        #         eval_feature = torch.Tensor(self.eval_log['feature']).cuda()
        #     else:
        #         eval_feature = torch.Tensor(self.eval_log['feature']).cpu()
                
        #     model.eval() 
        #     eval_predicts = model(eval_feature)
        #     if isinstance(eval_predicts,tuple):
        #         eval_predicts = eval_predicts[1]
        #         eval_predicts = eval_predicts.view(eval_predicts.size(0))
        #         eval_predicts = eval_predicts.cpu()
        #     else:
        #         eval_predicts = eval_predicts.view(eval_predicts.size(0))
        #         eval_predicts = eval_predicts.cpu()
        model.set_train(False)
        eval_feature = mindspore.Tensor(np.array(self.eval_log['feature'].tolist(),dtype=np.float32))
        eval_predicts = model(eval_feature).view(-1)
        # eval_predicts = eval_predicts.view(eval_predicts.size())
        eval_predicts = eval_predicts.asnumpy()
        print(eval_predicts.shape)
        return eval_predicts
    
    
    def _get_stat(self, eval_predicts):
        # build evaluation dataframe
        df_eval = pd.DataFrame([], columns=['qid', 'did', 'label', 'predict'])
        df_eval['qid'] = self.eval_log['qid']
        df_eval['did'] = self.eval_log['did']
        df_eval['label'] = self.eval_log['label']
        df_eval['predict'] = eval_predicts
        
        # df_eval = df_eval.reindex(np.random.permutation(df_eval.index))
        # give corresponding rank position for each query-doc pair
        df_eval['rank'] = df_eval.groupby('qid')['predict'].rank(method='first', ascending=False)
        # sort for each group
        temp = df_eval.groupby('qid').apply(lambda x: x.sort_values('rank', ascending=True))
        temp.reset_index(drop=True, inplace=True)
        # output click_list/ips_list/rank_list for each session
        temp_group = temp.groupby('qid')
        df_static = temp_group['label'].apply(list).reset_index()
        df_static.rename(columns={'label':'label_list'},inplace = True)
        df_static['rank_list'] = temp_group['rank'].apply(list).reset_index()['rank']
        
        return df_eval, df_static
    
class Test_Evaluator(Evaluator):
    
    def __init__(self, eval_log, eval_positions, use_cuda = False):
        super(Test_Evaluator, self).__init__(eval_log, eval_positions, use_cuda)
        self.MAP_evals = []
        self.NDCG_full_evals = []
        for p in eval_positions:
            self.MAP_evals.append(AP(p))
        for _ in eval_positions:
            self.NDCG_full_evals.append(NDCG(999))


    def evaluate(self, model):
        eval_predicts = super(Test_Evaluator,self)._get_predicts(model)
        df_eval, df_stat = super(Test_Evaluator,self)._get_stat(eval_predicts)
    
        #AUC = metrics.roc_auc_score(df_eval['label'], df_eval['predict'])
        NDCGs = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
                 for NDCG_eval in self.NDCG_evals]
        # ARPs = [df_stat['label_list'].apply(ARP_eval.evaluate).mean() 
        #              for ARP_eval in self.ARP_evals]
        MRRs = [df_stat['label_list'].apply(MRR_eval.evaluate).mean() 
                for MRR_eval in self.MRR_evals]
        MAPs = [df_stat['label_list'].apply(MAP_eval.evaluate).mean() 
                     for MAP_eval in self.MAP_evals]
        Precisions = [df_stat['label_list'].apply(Precision_eval.evaluate).mean() 
                        for Precision_eval in self.Precision_evals]
        NDCG_fulls = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
                        for NDCG_eval in self.NDCG_full_evals]       

        # wrap eval result into pandas dataframe
        eval_result = pd.DataFrame([],columns=['NDCG','MRR','Precision','MAP','NDCG_full'], index= self.eval_positions)
        eval_result['NDCG'] = NDCGs
        eval_result['MRR'] = MRRs
        eval_result['Precision'] = Precisions
        eval_result['MAP'] = MAPs
        eval_result['NDCG_full'] = NDCG_fulls

        return eval_result
        

# class Vali_Evaluator(Evaluator):
    
#     def __init__(self, eval_log, eval_positions, use_cuda = False, 
#                  pair_wise = False, pair_df = None, with_weight = False):
#         super(Vali_Evaluator, self).__init__(eval_log, eval_positions, use_cuda)
#         self.pair_wise = pair_wise
#         self.pair_df = pair_df
#         self.with_weight = with_weight
        
#     def evaluate(self, model, loss_type=None):
#         if loss_type==None:
#             raise TypeError('loss_type must be given!')
            
#         val_loss, eval_predicts = self._get_predicts(model, loss_type)
#         if self.with_weight == 'dr':
#             df_eval, df_stat = self._get_stat(eval_predicts, after_pretrain='dr')
#         elif self.with_weight == 'dm':
#             df_eval, df_stat = self._get_stat(eval_predicts, after_pretrain='dm')
#         else:
#             df_eval, df_stat = self._get_stat(eval_predicts)

        
#         if self.with_weight == 'ips' or self.with_weight == 'ips_sel':
#             # return metrics' values averaged on session num with ips weights
            
#             NDCGs = [df_stat.apply(lambda row: NDCG_eval.evaluate(targets = row['click_list'], weights = row['ips_list']), axis = 1).mean() 
#                      for NDCG_eval in self.NDCG_evals]
#             # NDCGs = [df_stat.apply(lambda row: NDCG_eval.evaluate(targets = list(np.array(row['click_list'])*np.array(row['ips_list']))), axis = 1).mean() 
#             #          for NDCG_eval in self.NDCG_evals]         
#             # ARPs = [df_stat.apply(lambda row: ARP_eval.evaluate(targets = row['click_list'], weights = row['ips_list']), axis = 1).mean() 
#             #          for ARP_eval in self.ARP_evals]
#             MRRs = [df_stat.apply(lambda row: MRR_eval.evaluate(targets = row['click_list'], weights = row['ips_list']), axis = 1).mean() 
#                      for MRR_eval in self.MRR_evals]
#             # MAPs = [df_stat.apply(lambda row: MAP_eval.evaluate(targets = row['click_list'], weights = row['ips_list']), axis = 1).mean() 
#             #         for MAP_eval in self.MAP_evals]
#             Precisions = [df_stat.apply(lambda row: Precision_eval.evaluate(targets = row['click_list'], weights = row['ips_list']), axis = 1).mean() 
#                             for Precision_eval in self.Precision_evals]
#             # AUC = metrics.roc_auc_score(df_eval['isClick'], df_eval['predict'], sample_weight=df_eval['ips_weight_for_train'])

#         elif self.with_weight == 'dr':
#             # return metrics' values averaged on session num with ips weights

#             NDCGs = [df_stat.apply(lambda row: NDCG_eval.evaluate(targets = row['dr_label_list'], weights = row['dr_score_list']), axis = 1).mean() 
#                      for NDCG_eval in self.NDCG_evals]
#             # ARPs = [df_stat.apply(lambda row: ARP_eval.evaluate(targets = row['dr_label_list'], weights = row['dr_score_list']), axis = 1).mean() 
#             #          for ARP_eval in self.ARP_evals]
#             MRRs = [df_stat.apply(lambda row: MRR_eval.evaluate(targets = row['dr_label_list'], weights = row['dr_score_list']), axis = 1).mean() 
#                      for MRR_eval in self.MRR_evals]
#             # MAPs = [df_stat.apply(lambda row: MAP_eval.evaluate(targets = row['dr_label_list'], weights = row['dr_score_list']), axis = 1).mean() 
#             #         for MAP_eval in self.MAP_evals]
#             Precisions = [df_stat.apply(lambda row: Precision_eval.evaluate(targets = row['dr_label_list'], weights = row['dr_score_list']), axis = 1).mean() 
#                             for Precision_eval in self.Precision_evals]

#             # AUC = 0
#             # AUC = metrics.roc_auc_score(df_eval['dr_label'], df_eval['predict'], sample_weight=df_eval['dr_score'])

#         elif self.with_weight == 'dm':
#             # return metrics' values averaged on session num with ips weights

#             NDCGs = [df_stat.apply(lambda row: NDCG_eval.evaluate(targets = row['dm_label_list'], weights = row['dm_score_list']), axis = 1).mean() 
#                      for NDCG_eval in self.NDCG_evals]
#             # ARPs = [df_stat.apply(lambda row: ARP_eval.evaluate(targets = row['dm_label_list'], weights = row['dm_score_list']), axis = 1).mean() 
#             #          for ARP_eval in self.ARP_evals]
#             MRRs = [df_stat.apply(lambda row: MRR_eval.evaluate(targets = row['dm_label_list'], weights = row['dm_score_list']), axis = 1).mean() 
#                      for MRR_eval in self.MRR_evals]
#             # MAPs = [df_stat.apply(lambda row: MAP_eval.evaluate(targets = row['dm_label_list'], weights = row['dm_score_list']), axis = 1).mean() 
#             #         for MAP_eval in self.MAP_evals]
#             Precisions = [df_stat.apply(lambda row: Precision_eval.evaluate(targets = row['dm_label_list'], weights = row['dm_score_list']), axis = 1).mean() 
#                             for Precision_eval in self.Precision_evals]

#             # AUC = 0
#             # AUC = metrics.roc_auc_score(df_eval['dr_label'], df_eval['predict'], sample_weight=df_eval['dr_score'])

#         elif self.with_weight == 'naive':
#             # return metrics' values averaged on session num without ips weights
#             NDCGs = [df_stat['click_list'].apply(NDCG_eval.evaluate).mean() 
#                      for NDCG_eval in self.NDCG_evals]
#             # ARPs = [df_stat['click_list'].apply(ARP_eval.evaluate).mean() 
#             #          for ARP_eval in self.ARP_evals]
#             MRRs = [df_stat['click_list'].apply(MRR_eval.evaluate).mean()
#                      for MRR_eval in self.MRR_evals]
#             # MAPs = [df_stat['click_list'].apply(MAP_eval.evaluate).mean() 
#             #         for MAP_eval in self.MAP_evals]
#             Precisions = [df_stat['click_list'].apply(Precision_eval.evaluate).mean() 
#                             for Precision_eval in self.Precision_evals]
#             # AUC = metrics.roc_auc_score(df_eval['isClick'], df_eval['predict'])
        
#         # wrap eval result into pandas dataframe
#         eval_result = pd.DataFrame([],columns=['NDCG','MRR','Precision'], index= self.eval_positions)
#         eval_result['NDCG'] = NDCGs
#         eval_result['MRR'] = MRRs
#         eval_result['Precision'] = Precisions
        
#         return val_loss, eval_result
    
#     def _get_predicts(self, model, loss_type):
#         with torch.no_grad():       
#             model.eval()
#             if self.use_cuda:
#                 eval_targets = torch.Tensor(self.eval_log['isClick']).cuda()
#                 eval_feature = torch.Tensor(self.eval_log['feature']).cuda()
#                 eval_weights = torch.Tensor(self.eval_log['ips_weight']).cuda()
#             else:
#                 eval_targets = torch.Tensor(self.eval_log['isClick']).cpu()
#                 eval_feature = torch.Tensor(self.eval_log['feature']).cpu()
#                 eval_weights = torch.Tensor(self.eval_log['ips_weight']).cpu()

#             # given predicts and loss on validation click log data
#             if self.pair_wise:
#                 if type(self.pair_df) != pd.core.frame.DataFrame:
#                     raise TypeError('pair dataframe must not be None!')
#                 else:
#                     if self.use_cuda:
#                         eval_posi = torch.Tensor(self.pair_df['pos_feature']).cuda()
#                         eval_neg = torch.Tensor(self.pair_df['neg_feature']).cuda()
#                         eval_diff = torch.Tensor(self.pair_df['rel_diff']).cuda()
#                         if self.with_weight=='naive':
#                             eval_diff = torch.Tensor(self.pair_df['click_diff']).cuda()
#                     else:
#                         eval_posi = torch.Tensor(self.pair_df['pos_feature']).cpu()
#                         eval_neg = torch.Tensor(self.pair_df['neg_feature']).cpu()
#                         eval_diff = torch.Tensor(self.pair_df['rel_diff']).cpu()
#                         if self.with_weight=='naive':
#                             eval_diff = torch.Tensor(self.pair_df['click_diff']).cpu()

#                     posi_pred = model(eval_posi)
#                     neg_pred = model(eval_neg)
#                     if isinstance(posi_pred,tuple):
#                         posi_pred = posi_pred[1]
#                     if isinstance(neg_pred,tuple):
#                         neg_pred = neg_pred[1]

#                     posi_pred = posi_pred.view(posi_pred.size(0))
#                     neg_pred = neg_pred.view(neg_pred.size(0))
#                     loss_func = loss_type(weight = eval_diff)
#                     val_loss = loss_func(posi_pred, neg_pred)

#                     eval_predicts = model(eval_feature)
#                     if isinstance(eval_predicts, tuple):
#                         eval_predicts = eval_predicts[1]
#                     eval_predicts = eval_predicts.view(eval_predicts.size(0))
#             else:
#                 eval_predicts = model(eval_feature)
#                 if isinstance(eval_predicts, tuple):
#                     eval_predicts = eval_predicts[1]
#                 else:
#                     pass
#                 # _, eval_predicts = model(eval_feature)
#                 # _, eval_predicts, _ = model(eval_feature)
#                 eval_predicts = eval_predicts.view(eval_predicts.size(0))
#                 loss_func = loss_type(weight = eval_weights)
#                 val_loss = loss_func(eval_predicts, eval_targets)

#             # trans to cpu data for using pandas
#             eval_predicts = eval_predicts.cpu()
#         return val_loss, eval_predicts
    
#     def _get_stat(self, eval_predicts, after_pretrain=None):
#         # build evaluation dataframe
#         df_eval = pd.DataFrame([], columns=[])
#         df_eval['sid'] = self.eval_log['sid']
#         df_eval['qid'] = self.eval_log['qid']
#         df_eval['did'] = self.eval_log['did']
#         df_eval['isClick'] = self.eval_log['isClick']
#         df_eval['ips_weight'] = self.eval_log['ips_weight']
#         df_eval['ips_weight_for_train'] = self.eval_log['ips_weight_for_train']
#         if after_pretrain == 'dr':
#             df_eval['dr_weight'] = self.eval_log['dr_weight']
#             df_eval['dr_score'] = self.eval_log['dr_score']
#             df_eval['dr_label'] = self.eval_log['dr_label']
#         elif after_pretrain == 'dm':
#             df_eval['dm_weight'] = self.eval_log['dm_weight']
#             df_eval['dm_score'] = self.eval_log['dm_score']
#             df_eval['dm_label'] = self.eval_log['dm_label']
#         df_eval['predict'] = eval_predicts.view(-1).tolist()

#         # df_eval = df_eval.reindex(np.random.permutation(df_eval.index))
#         # give corresponding rank position for each query-doc pair
#         df_eval = shuffle(df_eval)
#         df_eval['rank'] = df_eval.groupby('sid')['predict'].rank(method='first', ascending=False)
#         # sort for each group
#         temp = df_eval.groupby('sid').apply(lambda x: x.sort_values('rank', ascending=True))
#         temp.reset_index(drop=True, inplace=True)
#         # output click_list/ips_list/rank_list for each session
#         temp_group = temp.groupby(['qid','sid'])
#         df_static = temp_group['isClick'].apply(list).reset_index()
#         df_static.rename(columns={'isClick':'click_list'},inplace = True)
#         df_static['ips_list'] = temp_group['ips_weight'].apply(list).reset_index()['ips_weight']
#         if after_pretrain == 'dr':
#             df_static['dr_weight_list'] = temp_group['dr_weight'].apply(list).reset_index()['dr_weight']
#             df_static['dr_score_list'] = temp_group['dr_score'].apply(list).reset_index()['dr_score']
#             df_static['dr_label_list'] = temp_group['dr_label'].apply(list).reset_index()['dr_label']
#         if after_pretrain == 'dm':
#             df_static['dm_weight_list'] = temp_group['dm_weight'].apply(list).reset_index()['dm_weight']
#             df_static['dm_score_list'] = temp_group['dm_score'].apply(list).reset_index()['dm_score']
#             df_static['dm_label_list'] = temp_group['dm_label'].apply(list).reset_index()['dm_label']
#         df_static['rank_list'] = temp_group['rank'].apply(list).reset_index()['rank']
        
#         return df_eval, df_static
    
    
# class Vali_Fullinfo_Evaluator(Evaluator):
    
#     def __init__(self, eval_log, eval_positions, use_cuda = False, 
#                  pair_wise = False, pair_df = None):
#         super(Vali_Fullinfo_Evaluator, self).__init__(eval_log, eval_positions, use_cuda)
#         self.pair_wise = pair_wise
#         self.pair_df = pair_df
        
#     def evaluate(self, model, loss_type=None):
#         if loss_type==None:
#             raise TypeError('loss_type must be given!')
            
#         val_loss, eval_predicts = self._get_predicts(model, loss_type)
#         df_eval, df_stat = super(Vali_Fullinfo_Evaluator, self)._get_stat(eval_predicts)
        
#         # AUC = metrics.roc_auc_score(df_eval['label'], df_eval['predict'])
#         NDCGs = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
#                  for NDCG_eval in self.NDCG_evals]
#         # ARPs = [df_stat['label_list'].apply(ARP_eval.evaluate).mean() 
#         #              for ARP_eval in self.ARP_evals]
#         MRRs = [df_stat['label_list'].apply(MRR_eval.evaluate).mean() 
#                 for MRR_eval in self.MRR_evals]
#         # MAPs = [df_stat['label_list'].apply(MAP_eval.evaluate).mean() 
#         #              for MAP_eval in self.MAP_evals]
#         Precisions = [df_stat['label_list'].apply(Precision_eval.evaluate).mean() 
#                         for Precision_eval in self.Precision_evals]

#         # wrap eval result into pandas dataframe
#         eval_result = pd.DataFrame([],columns=['NDCG','MRR','Precision'], index= self.eval_positions)
#         eval_result['NDCG'] = NDCGs
#         eval_result['MRR'] = MRRs
#         eval_result['Precision'] = Precisions
        
#         return val_loss, eval_result
    
#     def _get_predicts(self, model, loss_type):
#         with torch.no_grad():       
#             model.eval()
#             if self.use_cuda:
#                 eval_targets = torch.Tensor(self.eval_log['label']).cuda()
#                 eval_feature = torch.Tensor(self.eval_log['feature']).cuda() 
#             else:
#                 eval_targets = torch.Tensor(self.eval_log['label']).cpu()
#                 eval_feature = torch.Tensor(self.eval_log['feature']).cpu()

#             # given predicts and loss on validation click log data
#             if self.pair_wise:
#                 if type(self.pair_df) != pd.core.frame.DataFrame:
#                     raise TypeError('pair dataframe must not be None!')
#                 else:
#                     if self.use_cuda:
#                         eval_posi = torch.Tensor(self.pair_df['pos_feature']).cuda()
#                         eval_neg = torch.Tensor(self.pair_df['neg_feature']).cuda()
#                         eval_diff = torch.Tensor(self.pair_df['rel_diff']).cuda()
#                     else:
#                         eval_posi = torch.Tensor(self.pair_df['pos_feature']).cpu()
#                         eval_neg = torch.Tensor(self.pair_df['neg_feature']).cpu()
#                         eval_diff = torch.Tensor(self.pair_df['rel_diff']).cpu()

#                     posi_pred = model(eval_posi)
#                     neg_pred = model(eval_neg)
#                     posi_pred = posi_pred.view(posi_pred.size(0))
#                     neg_pred = neg_pred.view(neg_pred.size(0))
#                     loss_func = loss_type(weight = eval_diff)
#                     val_loss = loss_func(posi_pred, neg_pred)

#                     eval_predicts = model(eval_feature)
#                     eval_predicts = eval_predicts.view(eval_predicts.size(0))
#             else:
#                 eval_predicts = model(eval_feature)
#                 eval_predicts = eval_predicts.view(eval_predicts.size(0))
#                 loss_func = loss_type()
#                 val_loss = loss_func(eval_predicts, eval_targets)

#             # trans to cpu data for using pandas
#             eval_predicts = eval_predicts.cpu()
#         return val_loss, eval_predicts


# class Init_Ranker_Evaluator(Evaluator):
    
#     def __init__(self, eval_log, eval_positions, use_cuda = False):
#         super(Init_Ranker_Evaluator, self).__init__(eval_log, eval_positions, use_cuda)
#         self.MAP_evals = []
#         self.NDCG_full_evals = []
#         for p in eval_positions:
#             self.MAP_evals.append(AP(p))
#         for _ in eval_positions:
#             self.NDCG_full_evals.append(NDCG(999))
        
#     def evaluate(self, predict_log):
#         eval_predicts = predict_log[0].values.tolist()
#         eval_predicts = torch.Tensor(eval_predicts)
#         df_eval, df_stat = super(Init_Ranker_Evaluator, self)._get_stat(eval_predicts)

#         # AUC = metrics.roc_auc_score(df_eval['label'], df_eval['predict'])
#         NDCGs = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
#                      for NDCG_eval in self.NDCG_evals]
#         # ARPs = [df_stat['label_list'].apply(ARP_eval.evaluate).mean() 
#         #              for ARP_eval in self.ARP_evals]
#         MRRs = [df_stat['label_list'].apply(MRR_eval.evaluate).mean() 
#                 for MRR_eval in self.MRR_evals]
#         MAPs = [df_stat['label_list'].apply(MAP_eval.evaluate).mean() 
#                      for MAP_eval in self.MAP_evals]
#         Precisions = [df_stat['label_list'].apply(Precision_eval.evaluate).mean() 
#                         for Precision_eval in self.Precision_evals]
#         NDCG_fulls = [df_stat['label_list'].apply(NDCG_eval.evaluate).mean() 
#                         for NDCG_eval in self.NDCG_full_evals] 

#         # wrap eval result into pandas dataframe
#         eval_result = pd.DataFrame([],columns=['NDCG','MRR','Precision','MAP','NDCG_full'], index= self.eval_positions)
#         eval_result['NDCG'] = NDCGs
#         eval_result['MRR'] = MRRs
#         eval_result['Precision'] = Precisions
#         eval_result['MAP'] = MAPs
#         eval_result['NDCG_full'] = NDCG_fulls

#         return eval_result


if __name__=="__main__":
    pass