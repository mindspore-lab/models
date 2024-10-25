from evaluator import TrafficStateEvaluator
from model import loss
from evaluator import eval_funcs
import pickle as pkl
import math
import numpy as np
import mindspore
import os

def transfer_dtype(y_true,y_pred):
    if isinstance(y_true,mindspore.Tensor):
        y_true=y_true.asnumpy()
    if isinstance(y_pred,mindspore.Tensor):
        y_pred=y_pred.asnumpy()
    return y_true.astype('float32'),y_pred.astype('float32')

def mask_mse_np(y_true,y_pred,region_mask,null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    
    Returns:
        np.float32 -- MSE
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0,1,0).astype('float32')
        label_mask = label_mask.reshape(-1,20,20)
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    mask_value=(y_true-y_pred).reshape(-1,20,20)
    return np.mean(((mask_value)*mask)**2)

def mask_rmse_np(y_true,y_pred,region_mask,null_val=None):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix, shape (W,H)
    
    Returns:
        np.float32 -- RMSE
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    return math.sqrt(mask_mse_np(y_true,y_pred,region_mask,null_val))

def nonzero_num(y_true):
    """get the grid number of have traffic accident in all time interval
    
    Arguments:
        y_true {np.array} -- shape:(samples,pre_len,W,H)
    Returns:
        {list} -- (samples,)
    """
    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list

def get_top(data,accident_nums):
    """get top-K risk grid
    Arguments:
        data {np.array} -- shape (samples,pre_len,W,H)
        accident_nums {list} -- (samples,)，grid number of have traffic accident in all time intervals
    Returns:
        {list} -- (samples,k)
    """
    data = data.reshape((data.shape[0],-1))
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(),key=lambda x:x[1],reverse=True)[:k]).keys()))
    return topk_list

def Recall(y_true,y_pred,region_mask):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    Returns:
        float -- recall
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    region_mask = np.where(region_mask >= 1,0,-1000)
    y_true=y_true.squeeze(-1)
    y_pred=y_pred.squeeze(-1)

    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)
    
    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)
    
    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums) * 100


def MAP(y_true,y_pred,region_mask):
    """
        y_true {np.ndarray} -- shape (samples,pre_len,W,H)
        y_pred {np.ndarray} -- shape (samples,pre_len,W,H)
        region_mask {np.ndarray} -- mask matrix,shape (W,H)
    """
    y_true,y_pred = transfer_dtype(y_true,y_pred)
    region_mask = np.where(region_mask >= 1,0,-1000)
    y_true=y_true.squeeze(-1)
    y_pred=y_pred.squeeze(-1)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask
    
    accident_grids_nums = nonzero_num(tmp_y_true)
    
    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]),list(pred_top_k[sample])))
    return sum(all_k_AP)/len(all_k_AP)

def AP(label_list,pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0




class TrafficAccidentEvaluator(TrafficStateEvaluator):

    def __init__(self, config):
        super(TrafficAccidentEvaluator, self).__init__(config)
        self.dataset=config['dataset']
        self.data_path = './raw_data/' + self.dataset + '/'
        self.data_path = os.path.join(self.data_path,'risk_mask.pkl')
        self.region_mask=pkl.load(open(self.data_path,'rb'))
        
    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        self.allowed_metrics = ["MAE", "MAPE", "MSE", "RMSE", "masked_MAE", "masked_MAPE", "masked_MSE", "masked_RMSE", "R2", "EVAR",
                                    "Precision", "Recall", "F1-Score", "MAP", "PCC"]
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficAccidentEvaluator'.format(str(metric)))

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
                batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                if metric + '@' + str(i) not in self.intermediate_result:
                    self.intermediate_result[metric + '@' + str(i)] = []
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_m(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_m(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_m(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_m(y_pred[:, :i], y_true[:, :i]).item())
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            mask_rmse_np(y_true[:, i - 1], y_pred[:, i - 1],self.region_mask))
                    elif metric == 'Recall':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            Recall(y_true[:, i - 1], y_pred[:, i - 1],self.region_mask))
                    elif metric == 'MAP':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            MAP(y_true[:, i - 1], y_pred[:, i - 1], self.region_mask))
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))
