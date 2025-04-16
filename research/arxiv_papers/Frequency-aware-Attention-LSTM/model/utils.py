import gc
import psutil
import logging
import numpy as np
import pywt
import pandas as pd

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def clear_memory():
    """清理内存并记录当前可用内存"""
    gc.collect()
    mem_info = psutil.virtual_memory()
    logging.info(f"当前可用内存: {mem_info.available / 1024 / 1024:.2f} MB")

def denoise_series(series):
    """处理Pandas Series的去噪"""
    if (series == 0).mean() > 0.8:
        return series.rolling(5, center=True, min_periods=1).max()
    else:
        try:
            coeffs = pywt.wavedec(series.values, 'sym5', level=4)
            sigma = np.median(np.abs(coeffs[-4])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(series)))
            coeffs[1:] = [pywt.threshold(c, uthresh, 'soft') for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, 'sym5')
            return pd.Series(denoised[:len(series)], index=series.index)
        except:
            return series.rolling(7, center=True, min_periods=1, win_type='gaussian').mean(std=2)