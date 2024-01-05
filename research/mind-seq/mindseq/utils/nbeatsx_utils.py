import mindspore as ms
import numpy as np
import pandas as pd
import multiprocessing as mp

from math import sqrt
from functools import partial

from scipy import stats
from scipy.stats.distributions import chi2

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'serif'

AVAILABLE_METRICS = ['mse', 'rmse', 'mape', 'smape', 'mase', 'rmsse', 'gwtest',
                     'mini_owa', 'pinball_loss']
FONTSIZE = 20

######################################################################
# SIGNIFICANCE TEST
######################################################################

def Newey_West(Z, n_lags):
    """ Newey-West HAC estimator
    Parameters
    ----------
    Z: (n, k) ndarray
    n_lags: int
        number of lags to consider as available information.

    Returns
    -------
    omega_hat: Newey-West HAC estimator of the covariance matrix
    """

    assert n_lags > 0

    n, k = Z.shape

    Z = Z - np.ones((n, 1)) * np.mean(Z, axis=0)
    gamma = -999 * np.ones((n_lags, k))
    omega_hat = (1/n) * np.matmul(np.transpose(Z), Z)

    Zlag = np.array([np.pad(Z, ((i,0), (0,0)), mode='constant', 
                            constant_values = 0)[:n] 
                     for i in range(1, n_lags + 1)])
    gamma = (1/n) * (np.matmul(np.transpose(Z), Zlag) + 
            np.matmul(np.einsum('ijk -> ikj', Zlag), Z))
    weights = 1 - np.array(range(1,n_lags + 1))/(n_lags + 1)
    omega_hat = omega_hat + \
                np.sum(gamma * np.expand_dims(weights, 
                                              axis = (1,2)), 
                                              axis = 0)
    return omega_hat

def GW_CPA_test(loss1: np.ndarray, 
                loss2: np.ndarray, 
                tau: int,
                alpha: float=0.05,
                conditional: bool=False,
                verbose: bool=True):
    """ 
    Giacomini-White Conditional Predictive Ability Test
    Parameters
    ----------
    loss1: numpy array
        losses of model 1
    loss2: numpy array
        losses of model 2
    tau: int
        the past information treated as 'available' for the test.
    unconditional: boolean, 
        True if unconditional (DM test), False if conditional (GW test).
    verbose: boolean, 
        True if prints of test are needed

    Returns
    -------
    test_stat: test statistic of the conditional predictive ability test
    crit_val: critical value of the chi-square test for a 5% confidence level
    p-vals: (k,) p-value of the test
    """   

    assert len(loss1) == len(loss2)

    lossdiff = loss1 - loss2
    t = len(loss1)
    instruments = np.ones_like(loss1)

    if conditional:
        instruments = np.hstack((instruments[:t-tau], 
                                 lossdiff[:-tau]))
        lossdiff = lossdiff[tau:]
        t = t - tau

    reg = instruments * lossdiff
    
    if tau == 1:
        res_beta = np.linalg.lstsq(reg, np.ones((t)), rcond=None)[0]

        err = np.ones((t,1)) - reg.dot(res_beta)
        r2 = 1 - np.mean(err**2)
        test_stat = t * r2
    
    else:

        zbar = np.mean(reg, axis=0)
        n_lags = tau - 1
        omega = Newey_West(Z=reg, n_lags=n_lags)
        test_stat = np.expand_dims(t*zbar, 
                                   axis=0).dot(np.linalg.inv(omega)).\
                                   dot(zbar)
      
    test_stat *= np.sign(np.mean(lossdiff))
    
    q = reg.shape[1]
    crit_val = chi2.ppf(1-alpha, df=q)
    p_val = 1 - chi2.cdf(test_stat, q)

    av_diff_loss = np.mean(loss1-loss2)
    s = '+' if np.mean(loss1-loss2) > 0 else '-'
    
    if verbose:
        if conditional: print('\nConditional test:\n')
        if not conditional: print('\nUnconditional test:\n')
        print(f'Forecast horizon: {tau}, Nominal Risk Level: {alpha}')
        print(f'Test-statistic: {test_stat} ({s})')
        print(f'Critical value: {crit_val}')
        print(f'p-value: {p_val}\n')
    
    return test_stat, crit_val, p_val

def gwtest(loss1, loss2, tau=1, conditional=1):
    d = loss1 - loss2
    TT = np.max(d.shape)

    if conditional:
        instruments = np.stack([np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
    else:
        instruments = np.ones_like(d)
        T = TT
    
    instruments = np.array(instruments, ndmin=2)

    reg = np.ones_like(instruments) * -999
    for jj in range(instruments.shape[0]):
        reg[jj, :] = instruments[jj, :] * d
    
    if tau == 1:
        # print(reg.shape, T)
        # print(reg.T)        
        betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
        err = np.ones((T, 1)) - np.dot(reg.T, betas)
        r2 = 1 - np.mean(err**2)
        GWstat = T * r2
    else:
        raise NotImplementedError
        zbar = np.mean(reg, -1)
        nlags = tau - 1
        # ...
    
    GWstat *= np.sign(np.mean(d))
    # pval = 1 - scipy.stats.norm.cdf(GWstat)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    # return pval
    
    q = reg.shape[0]
    pval = 1 - stats.chi2.cdf(GWstat, q)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    return pval

def get_nbeatsx_cmap():
    cmap = cm.get_cmap('pink', 512)
    yellows = cmap(np.linspace(0.5, 0.95, 256))

    cmap = cm.get_cmap('Blues', 256)
    blues = cmap(np.linspace(0.45, 0.75, 256))

    newcolors = np.concatenate([yellows, blues])

    #extra = np.array([116/256, 142/256, 157/256, 1])
    extra = np.array([66/256, 75/256, 98/256, 1])
    #extra = np.array([3/256, 34/256, 71/256, 1])
    newcolors[-10:, :] = extra
    newcmap = ListedColormap(newcolors)
    return newcmap

def get_epftoolbox_cmap():
    cmap = cm.get_cmap('YlGn_r', 512)
    #yellows = cmap(np.linspace(0.65, 1.0, 256))
    yellows = cmap(np.linspace(0.6, 1.0, 256))

    cmap = cm.get_cmap('gist_heat_r', 256)
    #reds = cmap(np.linspace(0.55, 0.75, 256))
    reds = cmap(np.linspace(0.39, 0.66, 256))

    newcolors = np.concatenate([yellows, reds])

    #extra = np.array([116/256, 142/256, 157/256, 1])
    #extra = np.array([66/256, 75/256, 98/256, 1])
    #extra = np.array([3/256, 34/256, 71/256, 1])
    extra = np.array([0, 0, 0, 1])
    newcolors[-10:, :] = extra
    newcmap = ListedColormap(newcolors)
    return newcmap

def plot_GW_test_pvals(pvals, labels, title):
    assert len(pvals)==len(labels), 'Wrong pvals and labels dimensions.'

    #plt.rc('text', usetex=True)
    plt.rc('axes', labelsize=FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONTSIZE)    # fontsize of the tick labels
    plt.rc('axes', titlesize=FONTSIZE+1)  # fontsize of the figure title

    fig = plt.figure(figsize=[6, 6])
    ax = plt.axes([.27, .22, .7, .7])

    data = np.array(np.float32(pvals))

    # Colormap with discontinuous limit
    #cmap = cm.get_cmap('GnBu', 256)
    #cmap = get_nbeatsx_cmap()
    cmap = get_epftoolbox_cmap()
    mappable = plt.imshow(data, cmap=cmap, vmin=0, vmax=0.1)

    ticklabels = labels #[r'$\textrm{' + e + '}$' for e in labels]
    plt.xticks(range(len(labels)), ticklabels, rotation=90., fontsize=FONTSIZE)
    plt.yticks(range(len(labels)), ticklabels, fontsize=FONTSIZE)

    plt.plot(list(range(len(labels))), 
             list(range(len(labels))), 'wx', c='white', markersize=FONTSIZE)
    plt.title(f'{title}', fontweight='bold', fontsize=FONTSIZE)

    # Turn spines off and create black grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1.5)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2) #0.05
    plt.colorbar(mappable, cax=cax)

    #fig.tight_layout()
    title = title.replace(" ", "_")
    title = title.replace(",", "")
    title = title.replace("(", "")
    title = title.replace(")", "")
    plt.savefig(f'./results/pvals/pvals_{title}.pdf', bbox_inches='tight')
    plt.show()

######################################################################
# METRICS
######################################################################

def mse(y, y_hat):
    """Calculates Mean Squared Error.

    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values

    Return
    ------
    scalar: MSE
    """
    mse = np.mean(np.square(y - y_hat))

    return mse

def rmse(y, y_hat):
    """Calculates Root Mean Squared Error.

    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: RMSE
    """
    rmse = sqrt(np.mean(np.square(y - y_hat)))

    return rmse

def mape(y, y_hat):
    """Calculates Mean Absolute Percentage Error.

    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: MAPE
    """
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
    mape = 100 * mape

    return mape

def smape(y, y_hat):
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: SMAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.mean(smape)
    assert smape <= 200, 'SMAPE should be lower than 200'

    return smape

def mae(y, y_hat, weights=None):
    """Calculates Mean Absolute Error.

    The mean absolute error 

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    weights: numpy array
        weights

    Return
    ------
    scalar: MAE
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'
    mae = np.average(np.abs(y - y_hat), weights=weights)
    return mae

def mase(y, y_hat, y_train, seasonality=1):
    """Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: MASE
    """
    scale = np.mean(abs(y_train[seasonality:] - y_train[:-seasonality]))
    mase = np.mean(abs(y - y_hat)) / scale
    mase = 100 * mase

    return mase

def rmae(y: np.ndarray,
         y_hat1: np.ndarray, y_hat2: np.ndarray,
         weights=None):
    """Calculates Relative Mean Absolute Error.

    The relative mean absolute error of two forecasts.
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat1: numpy array
      predicted values of first model
    y_hat2: numpy array
      predicted values of second model
    weights: numpy array
      weights for weigted average
    freq: int
      frequency of the y series, it will determine the
      seasonal naive benchmark

    Return
    ------
    scalar: rMAE
    """
    numerator = mae(y=y, y_hat=y_hat1, weights=weights)
    denominator = mae(y=y, y_hat=y_hat2, weights=weights)
    rmae = numerator/denominator
    return rmae

def rmsse(y, y_hat, y_train, seasonality=1):
    """Calculates the M5 Root Mean Squared Scaled Error.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: RMSSE
    """
    scale = np.mean(np.square(y_train[seasonality:] - y_train[:-seasonality]))
    rmsse = sqrt(mse(y, y_hat) / scale)
    rmsse = 100 * rmsse

    return rmsse

def mini_owa(y, y_hat, y_train, seasonality, y_bench):
    """Calculates the Overall Weighted Average for a single series.

    MASE, sMAPE for Naive2 and current model
    then calculatess Overall Weighted Average.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array of len h (forecasting horizon)
        predicted values
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y_train: numpy array
        insample values of the series for scale
    y_bench: numpy array of len h (forecasting horizon)
        predicted values of the benchmark model

    Return
    ------
    return: mini_OWA
    """
    mase_y = mase(y, y_hat, y_train, seasonality)
    mase_bench = mase(y, y_bench, y_train, seasonality)

    smape_y = smape(y, y_hat)
    smape_bench = smape(y, y_bench)

    mini_owa = ((mase_y/mase_bench) + (smape_y/smape_bench))/2

    return mini_owa

def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float=0.5, weights=None) -> np.ndarray:
    """Calculates the Pinball Loss.

    The Pinball loss measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for tau is 0.5 for the deviation from the median.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    tau: float
      Fixes the quantile against which the predictions are compared.
    Return
    ------
    return: pinball_loss
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'

    delta_y = y - y_hat
    pinball = np.maximum(tau * delta_y, (tau - 1) * delta_y)
    pinball = np.average(pinball, weights=weights) #pinball.mean()
    return pinball

def panel_mape(y_hat):
    y_hat = y_hat.copy()
    y_hat['mape'] = np.abs(y_hat['y_hat']-y_hat['y'])/np.abs(y_hat['y'])
    y_hat_grouped = y_hat.groupby('unique_id').mean().reset_index()
    mape = np.mean(y_hat_grouped['mape'])
    return mape

def panel_smape(y_hat):
    y_hat = y_hat.copy()
    y_hat['smape'] = np.abs(y_hat['y_hat']-y_hat['y'])/(np.abs(y_hat['y'])+np.abs(y_hat['y_hat']))
    y_hat_grouped = y_hat.groupby('unique_id').mean().reset_index()
    smape = 2 * np.mean(y_hat_grouped['smape'])
    return smape

def MAELoss(y, y_hat, mask=None):
    """MAE Loss

    Calculates Mean Absolute Error between
    y and y_hat. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in mindspore tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in mindspore tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mae:
    Mean absolute error.
    """
    mae = ms.ops.abs(y - y_hat) * mask
    mae = ms.ops.mean(mae)
    return mae


def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div

from collections import defaultdict

class TimeSeriesDataset:
    def __init__(self,
                 Y_df: pd.DataFrame,
                 X_df: pd.DataFrame=None,
                 S_df: pd.DataFrame=None,
                 f_cols: list=None,
                 ts_train_mask: list=None):
        assert type(Y_df) == pd.core.frame.DataFrame
        assert all([(col in Y_df) for col in ['unique_id', 'ds', 'y']])
        if X_df is not None:
            assert type(X_df) == pd.core.frame.DataFrame
            assert all([(col in X_df) for col in ['unique_id', 'ds']])

        print('Processing dataframes ...')
        # Pandas dataframes to data lists
        ts_data, s_data, self.meta_data, self.t_cols, self.X_cols = self._df_to_lists(Y_df=Y_df, S_df=S_df, X_df=X_df)

        # Dataset attributes
        self.n_series   = len(ts_data)
        self.max_len    = max([len(ts['y']) for ts in ts_data])
        self.n_channels = len(self.t_cols) # y, X_cols, insample_mask and outsample_mask
        self.frequency  = pd.infer_freq(Y_df.head()['ds'])
        self.f_cols     = f_cols

        # Number of X and S features
        self.n_x = 0 if X_df is None else len(self.X_cols)
        self.n_s = 0 if S_df is None else S_df.shape[1]-1 # -1 for unique_id

        print('Creating ts tensor ...')
        # Balances panel and creates
        # numpy  s_matrix of shape (n_series, n_s)
        # numpy ts_tensor of shape (n_series, n_channels, max_len) n_channels = y + X_cols + masks
        self.ts_tensor, self.s_matrix, self.len_series = self._create_tensor(ts_data, s_data)
        if ts_train_mask is None: ts_train_mask = np.ones(self.max_len)
        assert len(ts_train_mask)==self.max_len, f'Outsample mask must have {self.max_len} length'

        self._declare_outsample_train_mask(ts_train_mask)


    def _df_to_lists(self, Y_df, S_df, X_df):
        """
        """
        unique_ids = Y_df['unique_id'].unique()

        if X_df is not None:
            X_cols = [col for col in X_df.columns if col not in ['unique_id','ds']]
        else:
            X_cols = []

        if S_df is not None:
            S_cols = [col for col in S_df.columns if col not in ['unique_id']]
        else:
            S_cols = []

        ts_data = []
        s_data = []
        meta_data = []
        for i, u_id in enumerate(unique_ids):
            top_row = np.asscalar(Y_df['unique_id'].searchsorted(u_id, 'left'))
            bottom_row = np.asscalar(Y_df['unique_id'].searchsorted(u_id, 'right'))
            serie = Y_df[top_row:bottom_row]['y'].values
            last_ds_i = Y_df[top_row:bottom_row]['ds'].max()

            # Y values
            ts_data_i = {'y': serie}

            # X values
            for X_col in X_cols:
                serie =  X_df[top_row:bottom_row][X_col].values
                ts_data_i[X_col] = serie
            ts_data.append(ts_data_i)

            # S values
            s_data_i = defaultdict(list)
            for S_col in S_cols:
                s_data_i[S_col] = S_df.loc[S_df['unique_id']==u_id, S_col].values
            s_data.append(s_data_i)

            # Metadata
            meta_data_i = {'unique_id': u_id,
                           'last_ds': last_ds_i}
            meta_data.append(meta_data_i)

        t_cols = ['y'] + X_cols + ['insample_mask', 'outsample_mask']
    
        return ts_data, s_data, meta_data, t_cols, X_cols

    def _create_tensor(self, ts_data, s_data):
        """
        s_matrix of shape (n_series, n_s)
        ts_tensor of shape (n_series, n_channels, max_len) n_channels = y + X_cols + masks
        """
        s_matrix  = np.zeros((self.n_series, self.n_s))
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len))

        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()))
            
            ts_tensor[idx, :self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = ts_idx
            ts_tensor[idx,  self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = 1

            # To avoid sampling windows without inputs available to predict we shift -1
            # outsample_mask will be completed with the train_mask, this ensures available data
            ts_tensor[idx,  self.t_cols.index('outsample_mask'), -(ts_idx.shape[1]):] = 1
            s_matrix[idx, :] = list(s_data[idx].values())
            len_series.append(ts_idx.shape[1])
        return ts_tensor, s_matrix, np.array(len_series)

    def _declare_outsample_train_mask(self, ts_train_mask):
        # Update attribute and ts_tensor
        self.ts_train_mask = ts_train_mask

    def get_meta_data_col(self, col):
        col_values = [x[col] for x in self.meta_data]
        return col_values

    def get_filtered_ts_tensor(self, offset, output_size, window_sampling_limit, ts_idxs=None):
        last_outsample_ds = self.max_len - offset + output_size
        first_ds = max(last_outsample_ds - window_sampling_limit - output_size, 0)
        if ts_idxs is None:
            filtered_ts_tensor = self.ts_tensor[:, :, first_ds:last_outsample_ds]
        else:
            filtered_ts_tensor = self.ts_tensor[ts_idxs, :, first_ds:last_outsample_ds]
        right_padding = max(last_outsample_ds - self.max_len, 0) #To padd with zeros if there is "nothing" to the right
        ts_train_mask = self.ts_train_mask[first_ds:last_outsample_ds]

        assert np.sum(np.isnan(filtered_ts_tensor))<1.0, \
            f'The balanced balanced filtered_tensor has {np.sum(np.isnan(filtered_ts_tensor))} nan values'
        return filtered_ts_tensor, right_padding, ts_train_mask

    def get_f_idxs(self, cols):
        # Check if cols are available f_cols and return the idxs
        assert all(col in self.f_cols for col in cols), f'Some variables in {cols} are not available in f_cols.'
        f_idxs = [self.X_cols.index(col) for col in cols]
        return f_idxs

class TimeSeriesLoader(object):
    def __init__(self,
                 ts_dataset:TimeSeriesDataset,
                 model:str,
                 offset:int,
                 window_sampling_limit: int,
                 input_size: int,
                 output_size: int,
                 idx_to_sample_freq: int,
                 batch_size: int,
                 is_train_loader: bool,
                 shuffle:bool,
                 rank_id:int,
                 dist_flag:bool):
        """
        Time Series Loader object, used to sample time series from TimeSeriesDataset object.
        Parameters
        ----------
        ts_dataset: TimeSeriesDataset
        Time Series Dataet object which contains data in MindSpore tensors optimized for sampling.
        model: str ['nbeats']
            Model which will use the loader, affects the way of constructing batches.
        offset: int
            Equivalent to timestamps in test (data in test will not be sampled). It is used to filter
            the MindSpore tensor containing the time series, to avoid using the future during training.
        window_sampling_limit: int
            Equivalent to calibration window. Length of the history (prior to offset) which will be sampled
        input_size: int
            Size of inputs of each window (only for NBEATS), eg. 7 days
        ouput_size: int
            Forecasting horizon
        idx_to_sample_freq: int
            Frequency of sampling. Eg: 1 for data_augmentation, 24 for sampling only at 12:00am
        batch_size: int
            Number of batches (windows) to sample
        is_train_loader: bool
            True: will only sample time stamps with 1s in mask, False: will only sample time stamps with 0s in mask
        shuffle: bool
            Indicates if windows should be shuffled. True is used for training and False for predicting.
        """
        # Dataloader attributes
        self.model = model
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.idx_to_sample_freq = idx_to_sample_freq
        self.offset = offset
        self.ts_dataset = ts_dataset
        self.t_cols = self.ts_dataset.t_cols
        self.is_train_loader = is_train_loader # Boolean variable for train and validation mask
        self.shuffle = shuffle # Boolean to shuffle data, useful for validation

        # Create rolling window matrix in advanced for faster access to data and broadcasted s_matrix
        self.rank_id = rank_id
        self.dist_flag = dist_flag
        self._create_train_data()
        self.len_series = []

    def _update_sampling_windows_idxs(self):

        # Only sample during training windows with at least one active output mask and input mask
        outsample_condition = ms.ops.sum(self.ts_windows[:, self.t_cols.index('outsample_mask'), -self.output_size:], dim=1)
        insample_condition = ms.ops.sum(self.ts_windows[:, self.t_cols.index('insample_mask'), :self.input_size], dim=1)
        sampling_idx = ms.ops.nonzero(outsample_condition * insample_condition > 0) #element-wise product
        sampling_idx = list(sampling_idx.flatten().numpy())
        return sampling_idx

    def _create_windows_tensor(self):
        """
        Comment here
        TODO: Cuando creemos el otro dataloader, si es compatible lo hacemos funcion transform en utils
        """
        # Memory efficiency is gained from keeping across dataloaders common ts_tensor in dataset
        # Filter function is used to define train tensor and validation tensor with the offset
        # Default ts_idxs=ts_idxs sends all the data
        tensor, right_padding, train_mask = self.ts_dataset.get_filtered_ts_tensor(offset=self.offset, output_size=self.output_size,
                                                                                   window_sampling_limit=self.window_sampling_limit)
        tensor = ms.Tensor(tensor, ms.float32)
        train_mask = ms.Tensor(train_mask, ms.float32)

        # Outsample mask checks existance of values in ts, train_mask mask is used to filter out validation
        # is_train_loader inverts the train_mask in case the dataloader is in validation mode
        mask = train_mask if self.is_train_loader else (1 - train_mask)
        
        tensor[:, self.t_cols.index('outsample_mask'), :] = tensor[:, self.t_cols.index('outsample_mask'), :] * mask
        padder = ms.nn.ConstantPad1d(padding=(self.input_size, right_padding), value=0)
        tensor = padder(tensor)

        # Last output_size outsample_mask and y to 0
        tensor[:, self.t_cols.index('y'), -self.output_size:] = 0 # overkill to ensure no validation leakage
        tensor[:, self.t_cols.index('outsample_mask'), -self.output_size:] = 0

        # Creating rolling windows and 'flattens' them

        ### tensor.unfold
        # windows = tensor.unfold(dimension=-1, size=self.input_size + self.output_size, step=self.idx_to_sample_freq)
        sz = self.input_size + self.output_size
        stp = self.idx_to_sample_freq
        s1, s2, s3 = tensor.shape
        windows = ms.ops.unfold(tensor.unsqueeze(-1), kernel_size=(self.input_size + self.output_size, 1), stride=self.idx_to_sample_freq)
        windows = windows.reshape(s1, -1, self.input_size + self.output_size, windows.shape[-1]).swapaxes(-1, -2)
        # n_serie, n_channel, n_time, window_size -> n_serie, n_time, n_channel, window_size
        #print(f'n_serie, n_channel, n_time, window_size = {windows.shape}')
        windows = windows.permute(0,2,1,3)
        #print(f'n_serie, n_time, n_channel, window_size = {windows.shape}')
        windows = windows.reshape(-1, self.ts_dataset.n_channels, self.input_size + self.output_size)

        # Broadcast s_matrix: This works because unfold in windows_tensor, orders: time, serie
        s_matrix = self.ts_dataset.s_matrix.repeat(repeats=int(len(windows)/self.ts_dataset.n_series), axis=0)

        return windows, s_matrix

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.choice(a=self.windows_sampling_idx,
                                           size=len(self.windows_sampling_idx), replace=False)
        else:
            sample_idxs = np.array(self.windows_sampling_idx)
        
        assert len(sample_idxs)>0, 'Check the data as sample_idxs are empty'
        n_batches = int(np.ceil(len(sample_idxs) / self.batch_size)) # Must be multiple of batch_size for paralel gpu

        for idx in range(n_batches):
            ws_idxs = sample_idxs[(idx * self.batch_size) : (idx + 1) * self.batch_size]
            batch = self.__get_item__(index=ws_idxs)
            yield batch

    def __get_item__(self, index):
        if self.model == 'nbeats':
            return self._nbeats_batch(index)
        elif self.model == 'esrnn':
            assert 1<0, 'hacer esrnn'
        else:
            assert 1<0, 'error'

    def _nbeats_batch(self, index):
        # Access precomputed rolling window matrix (RAM intensive)
        if isinstance(index, np.ndarray):
            index = index.tolist()
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        insample_y = windows[:, self.t_cols.index('y'), :self.input_size]
        insample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), :self.input_size]
        insample_mask = windows[:, self.t_cols.index('insample_mask'), :self.input_size]

        outsample_y = windows[:, self.t_cols.index('y'), self.input_size:]
        outsample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), self.input_size:]
        outsample_mask = windows[:, self.t_cols.index('outsample_mask'), self.input_size:]


        batch = {'s_matrix': s_matrix,
                 'insample_y': insample_y, 'insample_x':insample_x, 'insample_mask':insample_mask,
                 'outsample_y': outsample_y, 'outsample_x':outsample_x, 'outsample_mask':outsample_mask}
        return batch

    def _create_train_data(self):
        # Create rolling window matrix for fast information retrieval
        self.ts_windows, self.s_matrix = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()
        if self.dist_flag is not None:
            block = len(self.windows_sampling_idx) // self.dist_flag
            self.windows_sampling_idx = self.windows_sampling_idx[self.rank_id*block : min((self.rank_id + 1) * block, len(self.windows_sampling_idx))]

    def update_offset(self, offset):
        if offset == self.offset:
            return # Avoid extra computation
        self.offset = offset
        self._create_train_data()

    def get_meta_data_col(self, col):
        return self.ts_dataset.get_meta_data_col(col)

    def get_n_variables(self):
        return self.ts_dataset.n_x, self.ts_dataset.n_s

    def get_n_series(self):
        return self.ts_dataset.n_series

    def get_max_len(self):
        return self.ts_dataset.max_len

    def get_n_channels(self):
        return self.ts_dataset.n_channels

    def get_X_cols(self):
        return self.ts_dataset.X_cols

    def get_frequency(self):
        return self.ts_dataset.frequency

__all__ = ['SOURCE_URL', 'NP', 'PJM', 'BE', 'FR', 'DE', 'EPFInfo', 'EPF']

from pandas.tseries.frequencies import to_offset

import logging
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell
def download_file(directory: Union[str, Path], source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.

    Parameters
    ----------
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = source_url.split('/')[-1]
    filepath = directory / filename

    # Streaming, so we can iterate over the response.
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(source_url, stream=True, headers=headers)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
            f.flush()
    t.close()

    if total_size != 0 and t.n != total_size:
        logger.error('ERROR, something went wrong downloading data')

    size = filepath.stat().st_size
    logger.info(f'Successfully downloaded {filename}, {size}, bytes.')

    if decompress:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(directory)

        logger.info(f'Successfully decompressed {filepath}')

# Cell
@dataclass
class Info:
    """
    Info Dataclass of datasets.
    Args:
        groups (Tuple): Tuple of str groups
        class_groups (Tuple): Tuple of dataclasses.
    """
    groups: Tuple[str]
    class_groups: Tuple[dataclass]

    def get_group(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __getitem__(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __iter__(self):
        for group in self.groups:
            yield group, self.get_group(group)


# Cell
@dataclass
class TimeSeriesDataclass:
    """
    Args:
        S (pd.DataFrame): DataFrame of static features of shape
            (n_time_series, n_features).
        X (pd.DataFrame): DataFrame of exogenous variables of shape
            (sum n_periods_i for i=1..n_time_series, n_exogenous).
        Y (pd.DataFrame): DataFrame of target variable of shape
            (sum n_periods_i for i=1..n_time_series, 1).
        idx_categorical_static (list, optional): List of categorical indexes
            of S.
        group (str, optional): Group name if applies.
            Example: 'Yearly'
    """
    S: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    idx_categorical_static: Optional[List] = None
    group: Union[str, List[str]] = None

# Cell
SOURCE_URL = 'https://sandbox.zenodo.org/api/files/da5b2c6f-8418-4550-a7d0-7f2497b40f1b/'

# Cell
@dataclass
class NP:
    test_date: str = '2016-12-27'
    name: str = 'NP'

@dataclass
class PJM:
    test_date: str = '2016-12-27'
    name: str = 'PJM'

@dataclass
class BE:
    test_date: str = '2015-01-04'
    name: str = 'BE'

@dataclass
class FR:
    test_date: str = '2015-01-04'
    name: str = 'FR'

@dataclass
class DE:
    test_date: str = '2016-01-04'
    name: str = 'DE'

# Cell
EPFInfo = Info(groups=('NP', 'PJM', 'BE', 'FR', 'DE'),
               class_groups=(NP, PJM, BE, FR, DE))

# Cell
class EPF:
    @staticmethod
    def load(directory: str,
             group: str) -> Tuple[pd.DataFrame,
                                  Optional[pd.DataFrame],
                                  Optional[pd.DataFrame]]:
        """
        Downloads and loads EPF data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'NP', 'PJM', 'BE', 'FR', 'DE'.
        """
        path = Path(directory) / 'epf' / 'datasets'
        EPF.download(directory)
        class_group = EPFInfo.get_group(group)
        file = path / f'{group}.csv'
        df = pd.read_csv(file)
        df.columns = ['ds', 'y'] + \
                     [f'Exogenous{i}' for i in range(1, len(df.columns) - 1)]
        df['unique_id'] = group
        df['ds'] = pd.to_datetime(df['ds'])
        df['week_day'] = df['ds'].dt.dayofweek
        dummies = pd.get_dummies(df['week_day'], prefix='day')
        df = pd.concat([df, dummies], axis=1)
        dummies_cols = [col for col in df if col.startswith('day')]
        Y = df.filter(items=['unique_id', 'ds', 'y'])
        X = df.filter(items=['unique_id', 'ds', 'Exogenous1', 'Exogenous2', 'week_day'] + \
                      dummies_cols)
        return Y, X, None

    @staticmethod
    def load_groups(directory: str,
                    groups: List[str]) -> Tuple[pd.DataFrame,
                                                Optional[pd.DataFrame],
                                                Optional[pd.DataFrame]]:
        """
        Downloads and loads panel of EPF data
        according of groups.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        groups: List[str]
            Group names.
            Allowed groups: 'NP', 'PJM', 'BE', 'FR', 'DE'.
        """
        Y = []
        X = []
        for group in groups:
            Y_df, X_df, S_df = EPF.load(directory=directory, group=group)
            Y.append(Y_df)
            X.append(X_df)

        Y = pd.concat(Y).sort_values(['unique_id', 'ds']).reset_index(drop=True)
        X = pd.concat(X).sort_values(['unique_id', 'ds']).reset_index(drop=True)

        S = Y[['unique_id']].drop_duplicates().reset_index(drop=True)
        dummies = pd.get_dummies(S['unique_id'], prefix='static')
        S = pd.concat([S, dummies], axis=1)

        return Y, X, S

    @staticmethod
    def download(directory: str) -> None:
        """Downloads EPF Dataset."""
        path = Path(directory) / 'epf' / 'datasets'
        if not path.exists():
            for group in EPFInfo.groups:
                download_file(path, SOURCE_URL + f'{group}.csv')