
# Configuration settings for Traffic Flow Prediction Model

# Data settings
DATA_PATH = "data/Traffic.csv"
WINDOW_SIZE = 24

# Model settings
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
MSTIM_OUT_CHANNELS = 32

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# MindSpore context settings
DEVICE_TARGET = "CPU"  # Change to "GPU" if GPU is available
