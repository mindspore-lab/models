import mindspore as ms
from mindspore import nn, ops
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class TimeSeriesModel(nn.Cell):
    def __init__(self, input_size, output_size, pred_steps, hidden_size=256, num_layers=2):
        super().__init__()
        self.pred_steps = pred_steps
        
        # 双路并行结构
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 合并特征后的全连接层
        self.fc = nn.SequentialCell([
            nn.Dense(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dense(64, output_size*pred_steps)
        ])

    def construct(self, x):
        # 双路特征提取
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        
        # 动态特征融合（直接拼接）
        combined = ops.cat([lstm_out[:, -1, :], gru_out[:, -1, :]], axis=-1)
        
        # 输出预测
        return self.fc(combined).view(x.shape[0], self.pred_steps, -1)

class TimeSeriesPredictor:
    def __init__(self, target_cols, pred_steps=3, lookback=7):
        self.scaler = StandardScaler()
        self.target_cols = target_cols
        self.pred_steps = pred_steps
        self.lookback = lookback
        self.num_features = None
        self.target_indices = None
        self.model = None

    def inverse_transform(self, data):
        dummy = np.zeros((data.shape[0], self.num_features))
        dummy[:, self.target_indices] = data
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, self.target_indices]

    def evaluate(self, X, y):
        self.model.set_train(False)
        predictions = self.model(X)
        preds_flat = predictions.reshape((-1, len(self.target_cols))).asnumpy()
        y_flat = y.reshape((-1, len(self.target_cols))).asnumpy()
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
        
        X_train = ms.Tensor(np.array(X[:split_train]), dtype=ms.float32)
        y_train = ms.Tensor(np.array(y[:split_train]), dtype=ms.float32)
        X_val = ms.Tensor(np.array(X[split_train:split_val]), dtype=ms.float32)
        y_val = ms.Tensor(np.array(y[split_train:split_val]), dtype=ms.float32)
        X_test = ms.Tensor(np.array(X[split_val:]), dtype=ms.float32)
        y_test = ms.Tensor(np.array(y[split_val:]), dtype=ms.float32)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, patience=10):
        input_size = X_train.shape[-1]
        output_size = len(self.target_cols)
        self.model = TimeSeriesModel(
            input_size=input_size,
            output_size=output_size,
            pred_steps=self.pred_steps,
            hidden_size=512,
            num_layers=1
        )
        
        criterion = nn.MSELoss()
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=0.00001, weight_decay=1e-4)
        
        def forward_fn(x, y):
            output = self.model(x)
            loss = criterion(output, y)
            return loss
        
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.set_train(True)
            loss_value, grads = grad_fn(X_train, y_train)
            optimizer(grads)
            
            # 验证步骤
            self.model.set_train(False)
            val_output = self.model(X_val)
            val_loss = criterion(val_output, y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ms.save_checkpoint(self.model, 'best_model.ckpt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            if epoch % 10 == 0:
                train_mse, _, _ = self.evaluate(X_train, y_train)
                val_mse, _, _ = self.evaluate(X_val, y_val)
                print(f'Epoch {epoch:03} | Train Loss: {loss_value.asnumpy():.4f} | Val Loss: {val_loss.asnumpy():.4f} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}')
        
        # 加载最佳模型
        param_dict = ms.load_checkpoint('best_model.ckpt')
        ms.load_param_into_net(self.model, param_dict)

    def predict(self, X):
        self.model.set_train(False)
        outputs = self.model(X)
        return outputs.asnumpy()

if __name__ == "__main__":
    target_columns = ['Vict Age']
    prediction_steps = 3
    
    predictor = TimeSeriesPredictor(
        target_cols=target_columns,
        pred_steps=prediction_steps,
        lookback=7
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_data('Sample_Data_2020.csv')
    predictor.train_model(X_train, y_train, X_val, y_val, epochs=100, patience=20)
    
    test_mse, test_mae, test_rmse = predictor.evaluate(X_test, y_test)
    print(f'Test Metrics - MSE: {test_mse:.3f}, MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}')
    
    test_input = ops.expand_dims(X_test[0], 0)
    prediction = predictor.predict(test_input)
    
    print(f"Predicted values for next {prediction_steps} steps:", 
          predictor.inverse_transform(prediction[0]))