
# Traffic Flow Prediction using MSTIM Model (MindSpore)

This repository implements a **multi-scale temporal information modeling (MSTIM)** based deep learning model for traffic flow prediction, developed using the **MindSpore** framework. The model combines LSTM, CNN, and attention-style temporal aggregation to capture both short-term and long-term patterns in traffic data.

## 📁 Project Structure

```
traffic_prediction_mstim/
├── data/                   # Contains the traffic dataset (e.g., Traffic.csv)
├── models/
│   ├── mstim.py            # MSTIM module definition
│   └── model.py            # TrafficFlowPredictor model
├── utils/
│   ├── preprocess.py       # Data preprocessing functions
│   ├── dataset.py          # Dataset wrapper for MindSpore
│   └── metrics.py          # Evaluation metric computation
├── train.py                # Main training and evaluation script
├── config.py               # (Optional) Configuration settings
├── requirements.txt        # Project dependencies
└── README.md               # Project description
```

## 📊 Dataset

We use the [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume), which contains hourly traffic volume data from a highway in Minnesota, USA. The dataset includes weather, date-time, and traffic volume features.

- Format: CSV
- Time range: 2012 to 2018
- Target: `traffic_volume` (normalized)

## 🚀 How to Run

1. **Install dependencies**  
   (We recommend using a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```

2. **Place dataset**  
   Download and place the CSV data file in the `data/` directory. You can rename it as `Traffic.csv` or modify the path in `train.py`.

3. **Train and Evaluate Model**

   ```bash
   python train.py
   ```

4. **Model Output**  
   - Checkpoint saved as `traffic_flow_predictor.ckpt`
   - Evaluation metrics: MAE, MSE, RMSE

## 🧠 Model Highlights

- **MSTIM Module**: Combines multiple 1D convolution layers with Bi-LSTM for multi-scale sequence modeling.
- **LSTM Backbone**: Learns temporal dependencies in the input sequence.
- **Custom Preprocessing**: Includes feature standardization and weather condition encoding.

## 📦 Dependencies

See [`requirements.txt`](./requirements.txt) for a complete list of required packages.

## 📄 License

MIT License

---

Created by your AI team using the MindSpore deep learning framework.
