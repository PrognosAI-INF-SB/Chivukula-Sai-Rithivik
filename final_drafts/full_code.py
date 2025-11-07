# ============================================================
# COMBINED END-TO-END RUL PREDICTION PIPELINE AND DASHBOARD
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\chsai\OneDrive\Desktop\infosys internship\dataset\CMaps"
DATA_DIR = os.path.join(BASE_DIR, "train_and_test")
RUL_DIR = os.path.join(BASE_DIR, "rul")
SEQUENCES_DIR = os.path.join(BASE_DIR, "mile stone1 results")
MODELS_DIR = os.path.join(BASE_DIR, "mile stone2 results")
LABELS_DIR = os.path.join(BASE_DIR, "milestone3_4_results")

os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

FD_DATASETS = [1, 2, 3, 4]
SEQUENCE_LENGTH = 30
WARNING_THRESHOLD = 50
CRITICAL_THRESHOLD = 20

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_fd_data(fd_num):
    index_cols = ['unit_id', 'time_cycles']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    col_names = index_cols + setting_cols + sensor_cols

    train = pd.read_csv(os.path.join(DATA_DIR, f"train_FD00{fd_num}.txt"), sep=' ', header=None)
    train.dropna(axis=1, inplace=True)
    train.columns = col_names

    test = pd.read_csv(os.path.join(DATA_DIR, f"test_FD00{fd_num}.txt"), sep=' ', header=None)
    test.dropna(axis=1, inplace=True)
    test.columns = col_names

    y_test_true = pd.read_csv(os.path.join(RUL_DIR, f"RUL_FD00{fd_num}.txt"), sep=' ', header=None)
    y_test_true.dropna(axis=1, inplace=True)
    y_test_true.columns = ['RUL']

    return train, test, y_test_true

def preprocess_fd_data(train, test):
    max_cycles = train.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    train = train.merge(max_cycles, on='unit_id', how='left')
    train['RUL'] = train['max_cycle'] - train['time_cycles']
    train.drop('max_cycle', axis=1, inplace=True)
    train['RUL'] = train['RUL'].clip(upper=125)

    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    train.drop(drop_sensors, axis=1, inplace=True)
    test.drop(drop_sensors, axis=1, inplace=True)

    sensor_features = [col for col in train.columns if col.startswith('s_')]
    for sensor in sensor_features:
        train[f'{sensor}_roll_mean'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
        train[f'{sensor}_roll_std'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
        test[f'{sensor}_roll_mean'] = test.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
        test[f'{sensor}_roll_std'] = test.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))

    feature_cols = [col for col in train.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return train, test, feature_cols

def create_sequences(data, seq_length, feature_cols):
    X, y = [], []
    for uid in data['unit_id'].unique():
        unit_data = data[data['unit_id'] == uid]
        features = unit_data[feature_cols].values
        rul = unit_data['RUL'].values
        for i in range(len(unit_data) - seq_length + 1):
            X.append(features[i:i+seq_length])
            y.append(rul[i+seq_length-1])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
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
    return model

def label_rul(rul):
    if rul <= CRITICAL_THRESHOLD:
        return "CRITICAL"
    elif rul <= WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "NORMAL"

# ============================================================
# PROCESS ALL DATASETS
# ============================================================

for fd in FD_DATASETS:
    print(f"\n=== Processing FD00{fd} ===")
    train_df, test_df, y_test_true = load_fd_data(fd)
    train_df, test_df, feature_cols = preprocess_fd_data(train_df, test_df)
    X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH, feature_cols)

    # Test sequences
    X_test = []
    for uid in test_df['unit_id'].unique():
        data_unit = test_df[test_df['unit_id']==uid][feature_cols].values
        if len(data_unit) >= SEQUENCE_LENGTH:
            X_test.append(data_unit[-SEQUENCE_LENGTH:])
        else:
            pad = np.zeros((SEQUENCE_LENGTH-len(data_unit), data_unit.shape[1]))
            X_test.append(np.vstack([pad, data_unit]))
    X_test = np.array(X_test)
    y_test = np.clip(y_test_true['RUL'].values, 0, 125)

    # Save sequences
    np.savez_compressed(os.path.join(SEQUENCES_DIR, f"FD00{fd}_sequences.npz"),
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # Build and train model
    model = build_lstm((SEQUENCE_LENGTH, len(feature_cols)))
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=64,
                        callbacks=[early_stop, reduce_lr], verbose=1)

    # Predict & evaluate
    y_pred = model.predict(X_test).flatten()
    y_pred = np.maximum(y_pred, 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"FD00{fd} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Assign labels
    labels = [label_rul(r) for r in y_pred]
    labels_df = pd.DataFrame({"Sample_Index": np.arange(len(y_test)),
                              "Actual_RUL": y_test,
                              "Predicted_RUL": y_pred,
                              "Condition_Label": labels})

    # Save model, metrics, labels, plots
    model.save(os.path.join(MODELS_DIR, f"fd00{fd}_model.h5"))
    pd.DataFrame({"FD_Set":[f"FD00{fd}"], "RMSE":[rmse], "MAE":[mae], "R2_Score":[r2]})\
        .to_csv(os.path.join(LABELS_DIR, f"metrics_fd{fd}.csv"), index=False)
    labels_df.to_csv(os.path.join(LABELS_DIR, f"rul_labels_fd{fd}.csv"), index=False)

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0,max(y_test)],[0,max(y_test)],'r--')
    plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title(f"FD00{fd} Predicted vs Actual")
    plt.grid(True)
    plt.savefig(os.path.join(LABELS_DIR, f"pred_vs_actual_fd{fd}.png"))
    plt.close()

# ============================================================
# DASHBOARD
# ============================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "RUL Prediction Dashboard"

def load_dashboard(fd_num):
    seq_file = os.path.join(SEQUENCES_DIR, f"FD00{fd_num}_sequences.npz")
    model_file = os.path.join(MODELS_DIR, f"fd00{fd_num}_model.h5")
    label_file = os.path.join(LABELS_DIR, f"rul_labels_fd{fd_num}.csv")
    data = np.load(seq_file)
    X_test = data["X_test"]
    y_test = data["y_test"]
    model = load_model(model_file, custom_objects={'mse': MeanSquaredError()})
    labels_df = pd.read_csv(label_file)
    return X_test, y_test, model, labels_df

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("RUL Prediction Dashboard", className="text-center mt-4 mb-4 text-primary"))),
    dbc.Row([
        dbc.Col([html.Label("Select Dataset", className="fw-bold"),
                 dcc.Dropdown(id='fd-selector', options=[{'label': f'FD00{i}', 'value': i} for i in FD_DATASETS],
                              value=1, clearable=False)], width=3),
        dbc.Col(html.Div(id='metrics-display'), width=9)
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
             dbc.Col(dcc.Graph(id='residual-hist'), width=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id='trend-plot'), width=12)]),
    dbc.Row([dbc.Col([html.H5("Labels from Dataset", className="fw-bold"), html.Div(id='labels-table')])]),
    dbc.Row([dbc.Col(html.Footer("Developed by Sai Rithivik | Infosys Internship 2025",
                                 className="text-center text-muted mt-3 mb-2"))])
], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

@app.callback(
    [Output('metrics-display','children'),
     Output('scatter-plot','figure'),
     Output('residual-hist','figure'),
     Output('trend-plot','figure'),
     Output('labels-table','children')],
    [Input('fd-selector','value')]
)
def update_dash(fd_num):
    X_test, y_test, model, labels_df = load_dashboard(fd_num)
    y_pred = model.predict(X_test).flatten()
    y_pred = np.maximum(y_pred, 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("RMSE"), html.H4(f"{rmse:.2f}")]), color="light"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("MAE"), html.H4(f"{mae:.2f}")]), color="light"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("R² Score"), html.H4(f"{r2:.3f}")]), color="light"), width=3)
    ], className="mb-3 justify-content-center")

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
    scatter_fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode='lines', line=dict(color='red', dash='dash'), name='Ideal'))
    scatter_fig.update_layout(title="Predicted vs Actual RUL", xaxis_title="Actual RUL", yaxis_title="Predicted RUL")

    residuals = y_test - y_pred
    residual_fig = go.Figure()
    residual_fig.add_trace(go.Histogram(x=residuals, nbinsx=40))
    residual_fig.update_layout(title="Residual Distribution", xaxis_title="Residual", yaxis_title="Count")

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(y=y_test[:100], mode='lines', name='Actual'))
    trend_fig.add_trace(go.Scatter(y=y_pred[:100], mode='lines', name='Predicted'))
    trend_fig.update_layout(title="RUL Trend (First 100 Samples)", xaxis_title="Sample Index", yaxis_title="RUL")

    table = dbc.Table.from_dataframe(labels_df.head(10), striped=True, bordered=True, hover=True, size='sm')
    return metrics, scatter_fig, residual_fig, trend_fig, table

if __name__ == '__main__':
    app.run(debug=True, port=8050)
