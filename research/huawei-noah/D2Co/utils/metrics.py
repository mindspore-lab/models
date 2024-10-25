import numpy as np
from sklearn.metrics import roc_auc_score
"""
Metrics:

Normalized Discounted Cumulative Gain@K

Average Relevance Position@K

Precision@K

Average Precision@K

"""

class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets, weights=None):
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        if weights is not None:
            ipw = self._get_weights(weights, min(self.k, len(gain)))
            return np.sum(np.divide(np.multiply(ipw, gain), discount))
        else:
            return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    def _get_weights(self, weights, k):
        return weights[:k]
    
    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n+1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):

    def __init__(self, k=10, gain_type='exp2'):
        super(NDCG, self).__init__(k, gain_type)

    def evaluate(self, targets, weights=None):
        targets = np.array(targets)
        dcg = super(NDCG, self).evaluate(targets, weights = weights)
        idx = np.argsort(targets)
        ideal = targets[idx][::-1]
        if weights:
            weights = np.array(weights)
            ideal_weights = weights[idx][::-1]
            idcg = super(NDCG, self).evaluate(ideal, weights = ideal_weights)
        else:
            idcg = super(NDCG, self).evaluate(ideal, weights = weights)
        if idcg == 0:
            return .0
        else:
            return dcg / idcg

class ARP(object):

    def __init__(self, k=10):
        self.k = k

    def evaluate(self, targets, weights=None):
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        if sum(gain) == 0:
            return 0.0
        else:
            if weights:
                ipw = self._get_weights(weights, min(self.k, len(gain)))
                return np.sum(np.multiply(np.multiply(gain, ipw), discount))/sum(gain)
            else:
                return np.sum(np.multiply(gain, discount))/sum(gain)

    def _get_gain(self, targets):
        t = targets[:self.k]
        t_binary = [1 if (score > 0) else 0 for score in t]
        return t_binary

    def _get_discount(self, k):
        self.discount = np.array([p + 1 for p in range(k)])
        return self.discount
    
    def _get_weights(self, weights, k):
        return weights[:k]


class MRR(object):

    def __init__(self, k=10):
        self.k = k
    
    def evaluate(self, targets, weights=None):
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        if sum(gain) == 0:
            return 0.0
        else:
            if weights is not None:
                ipw = self._get_weights(weights, min(self.k, len(gain)))
                idx = np.argwhere(np.array(gain)!=0)[0,0]
                return ipw[idx]*(1/(idx+1))
                #return np.sum(np.multiply(np.multiply(gain, ipw), discount))/sum(gain)
            else:
                return 1/(np.argwhere(np.array(gain)!=0)[0,0] + 1)
                #return np.sum(np.multiply(gain, discount))/sum(gain)

    def _get_gain(self, targets):
        # t = targets[:self.k]
        # MRR do not require top-k, it evaluate the whole ranking list
        t = targets
        t_binary = [1 if (score > 0) else 0 for score in t]
        return t_binary

    def _get_discount(self, k):
        self.discount = np.array([1/(p + 1) for p in range(k)])
        return self.discount
    
    def _get_weights(self, weights, k):
        return weights[:k]

    
class Precision(object):

    def __init__(self, k=10):
        self.k = k

    def evaluate(self, targets, weights=None):
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        if weights:
            ipw = self._get_weights(weights, min(self.k, len(gain)))
            return np.sum(np.multiply(np.multiply(ipw, gain), discount))
        else:
            return np.sum(np.multiply(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        t_binary = [1 if (score > 0) else 0 for score in t]
        #self.gain_len = len(t_binary)
        return t_binary

    def _get_discount(self, k):
        self.discount = [1/k for _ in range(k)]
        return self.discount
    
    def _get_weights(self, weights, k):
        return weights[:k]
    

class AP(Precision):
    def __init__(self, k=10):
        super(AP, self).__init__(k)

    def evaluate(self, targets, weights=None):
        #gain = super(AP, self)._get_gain(targets)
        self.k = len(targets)
        gain = self._get_binary_targets(targets)
        # gain = targets
        Precision_list = [super(AP, self).evaluate(targets[:position + 1], weights = weights) for position in range(len(gain))]
        Precision_list_mul_gain = np.multiply(gain, Precision_list)
        relevant_num = sum(self._get_binary_targets(targets))
        if relevant_num == 0:
            return 0
        else:
            return sum(Precision_list_mul_gain)/relevant_num

    def _get_binary_targets(self, targets):
        t_binary = [1 if (score > 0) else 0 for score in targets]
        return t_binary

if __name__ == "__main__":
    ndcg_at_6 = NDCG(6)
    #arp_at_6 = ARP(6)
    mrr_at_6 = MRR(6)
    p_at_6 = Precision(6)
    ap_at_6 = AP()

    weights = [0.33,0.30,0.29,0.18,0.11,0.11]
    #targets = [1,0,1,1]
    targets = [1,1,0,0]

    #weights = [0.999, 1.995, 3.016, 4.001, 5.005, 6.009, 135193]
    #targets = [3, 2, 0, 1, 0, 0, 0]
    #targets = [1, 1, 0, 1, 0, 0, 0]
    #targets = [1, 1, 0, 1]
    #targets = [0, 0, 0, 0, 0, 0, 0]

    print(ndcg_at_6.evaluate(targets))
    print(mrr_at_6.evaluate(targets))
    print(p_at_6.evaluate(targets))
    print([p_at_6.evaluate(targets[:position + 1]) for position in range(len(targets))])
    print(ap_at_6.evaluate(targets))
    print('')
    print('test with weight:')
    print(ndcg_at_6.evaluate(targets, weights = weights))
    print(mrr_at_6.evaluate(targets, weights = weights))
    print(p_at_6.evaluate(targets, weights = weights))
    print([p_at_6.evaluate(targets[:position + 1], weights = weights) for position in range(len(targets))])
    print(ap_at_6.evaluate(targets, weights = weights))