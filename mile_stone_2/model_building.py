import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: DATA LOADING
# =============================================================================
print("Loading data...")

# Column names
index_cols = ['unit_id', 'time_cycles']
setting_cols = ['setting_1', 'setting_2', 'setting_3']
sensor_cols = [f's_{i}' for i in range(1, 22)]
col_names = index_cols + setting_cols + sensor_cols

# Load training data
train = pd.read_csv("C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/train_and_test/train_FD004.txt", sep=' ', header=None)
train.dropna(axis=1, inplace=True)
train.columns = col_names

# Load test data
test = pd.read_csv("C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/train_and_test/test_FD004.txt", sep=' ', header=None)
test.dropna(axis=1, inplace=True)
test.columns = col_names

# Load true RUL for test
y_test_true = pd.read_csv("C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/rul/RUL_FD004.txt", sep=' ', header=None)
y_test_true.dropna(axis=1, inplace=True)
y_test_true.columns = ['RUL']

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# =============================================================================
# PART 2: PREPROCESSING
# =============================================================================
print("\nPreprocessing...")

# Add RUL to training data
max_cycles = train.groupby('unit_id')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']
train = train.merge(max_cycles, on='unit_id', how='left')
train['RUL'] = train['max_cycle'] - train['time_cycles']
train.drop('max_cycle', axis=1, inplace=True)
train['RUL'] = train['RUL'].clip(upper=125)

# Drop constant sensors
drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
train.drop(drop_sensors, axis=1, inplace=True)
test.drop(drop_sensors, axis=1, inplace=True)

# Get sensor columns
sensor_features = [col for col in train.columns if col.startswith('s_')]

# Add rolling features (window=5)
print("Adding rolling features...")
for sensor in sensor_features:
    train[f'{sensor}_roll_mean'] = train.groupby('unit_id')[sensor].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    train[f'{sensor}_roll_std'] = train.groupby('unit_id')[sensor].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    test[f'{sensor}_roll_mean'] = test.groupby('unit_id')[sensor].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    test[f'{sensor}_roll_std'] = test.groupby('unit_id')[sensor].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )

# Select features for scaling
feature_cols = [col for col in train.columns if col not in ['unit_id', 'time_cycles', 'RUL']]

# Scale features
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

print(f"Number of features: {len(feature_cols)}")

# =============================================================================
# PART 3: SEQUENCE GENERATION
# =============================================================================
print("\nGenerating sequences...")

sequence_length = 30

def create_sequences(data, seq_length, feature_columns):
    X, y = [], []
    for unit_id in data['unit_id'].unique():
        unit_data = data[data['unit_id'] == unit_id]
        features = unit_data[feature_columns].values
        rul_values = unit_data['RUL'].values
        
        for i in range(len(unit_data) - seq_length + 1):
            X.append(features[i:i+seq_length])
            y.append(rul_values[i+seq_length-1])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train, sequence_length, feature_cols)

# Test sequences
X_test = []
for unit_id in test['unit_id'].unique():
    unit_data = test[test['unit_id'] == unit_id][feature_cols].values
    if len(unit_data) >= sequence_length:
        X_test.append(unit_data[-sequence_length:])
    else:
        pad_length = sequence_length - len(unit_data)
        padded = np.vstack([np.zeros((pad_length, unit_data.shape[1])), unit_data])
        X_test.append(padded)

X_test = np.array(X_test)
y_test = np.clip(y_test_true['RUL'].values, 0, 125)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# =============================================================================
# PART 4: BUILD LSTM MODEL
# =============================================================================
print("\nBuilding model...")

model = keras.Sequential([
    layers.Input(shape=(sequence_length, len(feature_cols))),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# =============================================================================
# PART 5: TRAIN MODEL
# =============================================================================
print("\nTraining model...")

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================================
# PART 6: EVALUATE
# =============================================================================
print("\nEvaluating...")

y_pred = model.predict(X_test).flatten()
y_pred = np.maximum(y_pred, 0)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("Evaluation Results")
print("="*50)
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# =============================================================================
# PART 7: SAVE PLOTS
# =============================================================================
# =============================================================================
# PART 7: SAVE PLOTS
# =============================================================================
print("\nSaving plots...")

# Specify your custom folder path
save_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone2 results"

# Create folder if it doesn’t exist
os.makedirs(save_dir, exist_ok=True)

# Training history plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=300)
plt.close()

# Predictions vs Actual plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Perfect Prediction')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.legend()
plt.title('Predictions vs Actual')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "predictions_vs_actual.png"), dpi=300)
plt.close()

print(f"Plots saved to: {save_dir}")
