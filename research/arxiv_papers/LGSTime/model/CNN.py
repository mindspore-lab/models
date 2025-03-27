import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class TimeSeriesModel(nn.Cell):
    def __init__(self, input_size, output_size, pred_steps, hidden_size=64, dropout=0.2):
        super(TimeSeriesModel, self).__init__()
        self.pred_steps = pred_steps
        
        # 修正后的CNN架构（使用整数padding）
        self.conv_layers = nn.SequentialCell(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1,  # 手动计算保持维度
                pad_mode='pad',  # 明确指定填充模式
                has_bias=True
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size*2,
                kernel_size=3,
                padding=1,  # 保持维度一致
                pad_mode='pad',
                has_bias=True
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应全局平均池化
            nn.Dropout(p=dropout)
        )
        
        # 全连接层
        self.fc = nn.SequentialCell(
            nn.Dense(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(64, output_size*pred_steps)
        )
        self.reshape = ops.Reshape()

    def construct(self, x):
        batch_size = x.shape[0]
        
        # 维度调整：(batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(0, 2, 1)
        
        # CNN处理
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(batch_size, -1)  # 展平
        
        # 全连接预测
        prediction = self.fc(conv_out)
        return self.reshape(prediction, (batch_size, self.pred_steps, -1))
        
class TimeSeriesPredictor:
    def __init__(self, target_cols, pred_steps=3, lookback=7):
        self.scaler = StandardScaler()
        self.target_cols = target_cols
        self.pred_steps = pred_steps
        self.lookback = lookback
        self.num_features = None
        self.target_indices = None

    def inverse_transform(self, data):
        dummy = np.zeros((data.shape[0], self.num_features))
        dummy[:, self.target_indices] = data
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, self.target_indices]

    def evaluate(self, X, y):
        X = Tensor(X, mindspore.float32)
        predictions = self.model(X)
        preds_flat = predictions.view((-1, len(self.target_cols))).asnumpy()
        y_flat = y.view((-1, len(self.target_cols))).asnumpy()
        mse = mean_squared_error(y_flat, preds_flat)
        mae = mean_absolute_error(y_flat, preds_flat)
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    def prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        self.num_features = df.shape[1]
        self.target_indices = df.columns.get_indexer(self.target_cols).tolist()
        scaled_data = self.scaler.fit_transform(df)
        X, y = [], []
        for i in range(len(scaled_data)-self.lookback-self.pred_steps):
            X.append(scaled_data[i:i+self.lookback])
            y.append(scaled_data[i+self.lookback:i+self.lookback+self.pred_steps, self.target_indices])
        
        total_samples = len(X)
        split_train = int(total_samples * 0.7)
        split_val = split_train + int(total_samples * 0.1)
        split_val = min(split_val, total_samples)
        
        X_train = Tensor(np.array(X[:split_train]), mindspore.float32)
        y_train = Tensor(np.array(y[:split_train]), mindspore.float32)
        X_val = Tensor(np.array(X[split_train:split_val]), mindspore.float32)
        y_val = Tensor(np.array(y[split_train:split_val]), mindspore.float32)
        X_test = Tensor(np.array(X[split_val:]), mindspore.float32)
        y_test = Tensor(np.array(y[split_val:]), mindspore.float32)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, patience=10):
        input_size = X_train.shape[-1]
        output_size = len(self.target_cols)
        
        # 初始化CNN模型
        self.model = TimeSeriesModel(
            input_size=input_size,
            output_size=output_size,
            pred_steps=self.pred_steps,
            hidden_size=512,
            dropout=0.3
        )
        
        criterion = nn.MSELoss()
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=0.001, weight_decay=1e-4)
        self.train_net = nn.TrainOneStepCell(nn.WithLossCell(self.model, criterion), optimizer)
        self.train_net.set_train()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练步骤
            loss = self.train_net(X_train, y_train)
            
            # 验证步骤
            self.model.set_train(False)
            val_pred = self.model(X_val)
            val_loss = criterion(val_pred, y_val)
            self.model.set_train(True)
            
            # 早停机制
            if val_loss.asnumpy() < best_val_loss:
                best_val_loss = val_loss.asnumpy()
                patience_counter = 0
                mindspore.save_checkpoint(self.model, 'best_model.ckpt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            # 每10轮打印训练进度
            if epoch % 10 == 0:
                train_mse, _, _ = self.evaluate(X_train, y_train)
                val_mse, _, _ = self.evaluate(X_val, y_val)
                print(f'Epoch {epoch:03} | Train Loss: {loss.asnumpy():.4f} | Val Loss: {val_loss.asnumpy():.4f} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}')
        
        # 加载最佳模型
        param_dict = mindspore.load_checkpoint('best_model.ckpt')
        mindspore.load_param_into_net(self.model, param_dict)

    def predict(self, X):
        self.model.set_train(False)
        prediction = self.model(Tensor(X, mindspore.float32))
        return prediction.asnumpy()

if __name__ == "__main__":
    target_columns = ['Vict Age']
    prediction_steps = 1
    
    predictor = TimeSeriesPredictor(
        target_cols=target_columns,
        pred_steps=prediction_steps,
        lookback=7
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_data('Sample_Data_2023.csv')
    
    predictor.train_model(X_train, y_train, X_val, y_val, epochs=100, patience=20)
    
    test_mse, test_mae, test_rmse = predictor.evaluate(X_test, y_test)
    print(f'Test Metrics - MSE: {test_mse:.3f}, MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}')
    
    test_input = X_test[0:1]
    prediction = predictor.predict(test_input)
    print(f"Predicted values for next {prediction_steps} steps:", 
          predictor.inverse_transform(prediction[0]))