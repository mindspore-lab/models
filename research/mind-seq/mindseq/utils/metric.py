import numpy as np
import mindspore as ms
def absolute_error(pred, label, missing_value=0):
    return ((pred - label) * get_valid(label, missing_value)).abs().sum()


def square_error(pred, label, missing_value=0):
    return (((pred - label) ** 2) * get_valid(label, missing_value)).sum()


def absolute_percentage_error(pred, label, missing_value=0):
    return (((pred - label) / (label + 1e-8)) * get_valid(label, missing_value)).abs().sum()


def get_valid(label, missing_value=0):
    return ((label - missing_value).abs() > 1e-8).to(dtype=ms.float32)


def num_valid(label, missing_value=0):
    return get_valid(label, missing_value).sum()


def masked_mae(pred, label, missing_value=0):
    return absolute_error(pred, label, missing_value) / (num_valid(label, missing_value) + 1e-8)


class Metric:
    @staticmethod
    def create_metric(name):
        if name == 'mae': return MetricMAE()
        if name == 'rmse': return MetricRMSE()
        if name == 'mse': return MetricMSE()
        if name == 'mape': return MetricMAPE()
        return None

    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.value = 0

    def update(self, pred, label):
        raise NotImplementedError("To be implemented")

    def get_value(self):
        raise NotImplementedError("To be implemented")


class MetricMAE(Metric):
    def __init__(self):
        super(MetricMAE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += absolute_error(pred, label)

    def get_value(self):
        return (self.value / (self.cnt + 1e-8)).asnumpy().item()


class MetricRMSE(Metric):
    def __init__(self):
        super(MetricRMSE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += square_error(pred, label)

    def get_value(self):
        return ms.ops.sqrt(self.value / (self.cnt + 1e-8)).asnumpy().item()


class MetricMSE(Metric):
    def __init__(self):
        super(MetricMSE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += square_error(pred, label)

    def get_value(self):
        return (self.value / (self.cnt + 1e-8)).asnumpy().item()


class MetricMAPE(Metric):
    def __init__(self):
        super(MetricMAPE, self).__init__()

    def update(self, pred, label):
        self.cnt += num_valid(label)
        self.value += absolute_percentage_error(pred, label)

    def get_value(self):
        return (self.value / (self.cnt + 1e-8)).asnumpy().item()


class Metrics:
    def __init__(self, metric_list, metric_index):
        self.metric_all = {m: Metric.create_metric(m) for m in metric_list} # create mae, rmse, mape metrics
        self.metric_horizon = {'{}-horizon'.format(m): [Metric.create_metric(m) for i in metric_index] for m in metric_list} # create different horizon metrics(3,6,12)
        self.metric_index = metric_index

    def reset(self):
        for m in self.metric_all.values(): m.reset()
        for k, arr in self.metric_horizon.items():
            for m in arr:
                m.reset()

    def update(self, pred, label):
        # pred/label: [batch_size, pred_len, num_nodes, dim]
        for m in self.metric_all.values(): m.update(pred, label)
        for k, arr in self.metric_horizon.items():
            for i, idx in enumerate(self.metric_index):
                arr[i].update(pred[:, :idx, :, :], label[:, :idx, :, :])

    def get_value(self):
        ret = {k: np.array([m.get_value()]) for k, m in self.metric_all.items()}
        for k, arr in self.metric_horizon.items():
            ret[k] = np.array([m.get_value() for m in arr])
        return ret

    def __repr__(self):
        out_str = []
        for k, v in sorted(self.get_value().items()):
            out_str += ['{}: {}'.format(k, v)]
        return '\t'.join(out_str)
