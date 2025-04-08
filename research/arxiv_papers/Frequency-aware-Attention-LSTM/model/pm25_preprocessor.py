import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import denoise_series
import logging


class PM25Preprocessor:
    """PM2.5数据预处理类"""

    def __init__(self, window_size=24):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.wind_columns = ['wind_SE', 'wind_cv', 'wind_NE', 'wind_NW', 'wind_other']

    def _process_features(self, df):
        """特征工程处理"""
        # 创建副本以避免修改原始数据
        df = df.copy()

        # 时间特征
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['season'] = df.index.month % 12 // 3

        # 气象交互特征
        df['temp_dew_diff'] = df['TEMP'] - df['DEWP']
        df['pressure_grad'] = df['PRES'].diff().abs().fillna(0)
        df['precip_3h'] = df['precipitation'].rolling(3, min_periods=1).sum()
        df['wind_humidity'] = df['Iws'] * df['HUMI'] / 100

        # 风向编码处理
        if 'cbwd' in df.columns:
            df = pd.get_dummies(df, columns=['cbwd'], prefix='wind', dummy_na=True)
            for wd in self.wind_columns + ['wind_nan']:
                if wd not in df.columns:
                    df[wd] = 0
        return df[self.wind_columns + [c for c in df.columns if c not in self.wind_columns]]

    def transform(self, data_path):
        """主转换方法"""
        df = pd.read_csv(data_path)
        logging.info(f"原始数据列名: {df.columns.tolist()}")

        # 时间处理
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime').sort_index()

        # 数据清洗
        if 'No' in df.columns:
            df = df.drop('No', axis=1)

        # PM数据合并
        pm_cols = [c for c in ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post'] if c in df.columns]
        if pm_cols:
            df['PM'] = df[pm_cols].mean(axis=1)
        else:
            raise ValueError("数据中未找到PM监测列")

        # 气象数据处理
        met_cols = ['DEWP', 'TEMP', 'HUMI', 'PRES', 'Iws', 'precipitation', 'Iprec']
        df[met_cols] = df[met_cols].interpolate(method='linear', limit_direction='both')
        df['precipitation'] = df['precipitation'].clip(upper=50)

        # 特征工程
        df = self._process_features(df)

        # 去噪处理
        for col in ['PM', 'TEMP', 'Iws', 'precipitation']:
            if col in df.columns:
                df[col] = denoise_series(df[col])

        # 划分数据集后进行标准化
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()  # 显式创建副本
        val_df = df.iloc[split_idx:].copy()  # 显式创建副本

        numeric_cols = [c for c in df.columns if c not in ['PM'] + self.wind_columns]

        # 确保所有数值列都是浮点类型
        for col in numeric_cols:
            if train_df[col].dtype in [np.int64, np.int32, bool]:
                train_df[col] = train_df[col].astype(np.float64)
                val_df[col] = val_df[col].astype(np.float64)

        # 标准化处理
        train_df.loc[:, numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
        val_df.loc[:, numeric_cols] = self.scaler.transform(val_df[numeric_cols])

        self.feature_columns = df.columns.drop('PM')

        # 生成序列数据
        def create_sequences(dataframe):
            seq_data = []
            for i in range(len(dataframe) - self.window_size):
                window = dataframe.iloc[i:i + self.window_size]
                seq_data.append(window.values.astype(np.float32))
            return np.array(seq_data)

        return create_sequences(train_df), create_sequences(val_df)