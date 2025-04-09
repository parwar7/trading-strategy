import lightgbm as lgb
import pandas as pd
import numpy as np

# Generate dummy data
np.random.seed(42)
X = pd.DataFrame({
    'rsi': np.random.rand(100),
    'macd': np.random.rand(100),
    'boll_z': np.random.rand(100),
    'returns': np.random.rand(100),
})
y = (X['rsi'] > 0.5).astype(int)  # Dummy binary labels

# Train LightGBM model
train_data = lgb.Dataset(X, label=y)
params = {'objective': 'binary', 'metric': 'binary_logloss'}
model = lgb.train(params, train_data, num_boost_round=100)

# Save the model
model.save_model('model.txt')
print("Model saved as model.txt")
