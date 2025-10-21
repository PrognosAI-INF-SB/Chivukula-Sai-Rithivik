import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# PATHS
# =============================================================================
data_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/"
save_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone1 results"
os.makedirs(save_dir, exist_ok=True)

# =============================================================================
# PART 1: LOAD DATA
# =============================================================================
index_cols = ['unit_id', 'time_cycles']
setting_cols = ['setting_1', 'setting_2', 'setting_3']
sensor_cols = [f's_{i}' for i in range(1, 22)]
col_names = index_cols + setting_cols + sensor_cols

train = pd.read_csv(os.path.join(data_dir, "train_and_test/train_FD004.txt"), sep=' ', header=None)
train.dropna(axis=1, inplace=True)
train.columns = col_names

test = pd.read_csv(os.path.join(data_dir, "train_and_test/test_FD004.txt"), sep=' ', header=None)
test.dropna(axis=1, inplace=True)
test.columns = col_names

y_test_true = pd.read_csv(os.path.join(data_dir, "rul/RUL_FD004.txt"), sep=' ', header=None)
y_test_true.dropna(axis=1, inplace=True)
y_test_true.columns = ['RUL']

# =============================================================================
# PART 2: PREPROCESSING
# =============================================================================
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

# Sensor columns after dropping
sensor_features = [col for col in train.columns if col.startswith('s_')]

# Add rolling mean and std features (window=5)
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

# Scale features
feature_cols = [col for col in train.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# =============================================================================
# SAVE PREPROCESSED DATA (compressed CSV) FOR BOTH TRAIN AND TEST
# =============================================================================
train.to_csv(os.path.join(save_dir, "train_preprocessed_scaled_FD004.csv.gz"), index=False, compression='gzip')
test.to_csv(os.path.join(save_dir, "test_preprocessed_scaled_FD004.csv.gz"), index=False, compression='gzip')
print(f"Preprocessed train and test data saved as compressed CSV in: {save_dir}")

# =============================================================================
# PART 3: SEQUENCE GENERATION
# =============================================================================
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

# =============================================================================
# SAVE SEQUENCES AS SINGLE NPZ
# =============================================================================
np.savez_compressed(
    os.path.join(save_dir, "FD004_sequences.npz"),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
print("Sequences saved as FD004_sequences.npz in:", save_dir)
