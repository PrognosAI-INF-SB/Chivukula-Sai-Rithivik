import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = r"C:\Users\chsai\OneDrive\Desktop\infosys internship\dataset\CMaps"
SEQUENCES_DIR = os.path.join(BASE_DIR, "mile stone1 results")
MODELS_DIR = os.path.join(BASE_DIR, "mile stone2 results")
LABELS_BASE_DIR = os.path.join(BASE_DIR, "milestone 3&4 results")

FD_DATASETS = [1, 2, 3, 4]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_data_and_model(fd_num):
    """Load test sequences and model for given FD dataset."""
    try:
        seq_path = os.path.join(SEQUENCES_DIR, f"FD00{fd_num}_sequences.npz")
        model_path = os.path.join(MODELS_DIR, f"fd00{fd_num}_model.h5")

        data = np.load(seq_path)
        X_test = data["X_test"]
        y_test = data["y_test"]

        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        return X_test, y_test, model
    except Exception as e:
        print(f"Error loading FD00{fd_num}: {e}")
        return None, None, None


def load_labels(fd_num):
    """Load label CSV for given dataset."""
    try:
        folder_name = f"FD00{fd_num}"
        file_name = f"rul_labels_fd{fd_num}.csv"
        csv_path = os.path.join(LABELS_BASE_DIR, folder_name, file_name)

        df_labels = pd.read_csv(csv_path)
        return df_labels
    except Exception as e:
        print(f"Error loading labels for FD00{fd_num}: {e}")
        return pd.DataFrame({"Info": ["No label data available."]})


def predict_and_evaluate(X_test, y_test, model):
    """Predict RUL and compute evaluation metrics."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, r2

# =============================================================================
# DASH APP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "RUL Prediction Dashboard"

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("RUL Prediction Dashboard", 
                        className="text-center mt-4 mb-4 text-primary"))
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Dataset", className="fw-bold"),
            dcc.Dropdown(
                id='fd-selector',
                options=[{'label': f'FD00{i}', 'value': i} for i in FD_DATASETS],
                value=1,
                clearable=False
            )
        ], width=3),
        dbc.Col(html.Div(id='metrics-display'), width=9)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
        dbc.Col(dcc.Graph(id='residual-hist'), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='trend-plot'), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Labels from Dataset", className="fw-bold"),
            html.Div(id='labels-table')
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.Footer("Developed by Sai Rithivik | Infosys Internship 2025",
                            className="text-center text-muted mt-3 mb-2"))
    ])
], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

# =============================================================================
# CALLBACK
# =============================================================================
@app.callback(
    [Output('metrics-display', 'children'),
     Output('scatter-plot', 'figure'),
     Output('residual-hist', 'figure'),
     Output('trend-plot', 'figure'),
     Output('labels-table', 'children')],
    [Input('fd-selector', 'value')]
)
def update_dashboard(fd_num):
    X_test, y_test, model = load_data_and_model(fd_num)
    if X_test is None:
        return "Error loading data.", go.Figure(), go.Figure(), go.Figure(), "No data available."

    y_pred, rmse, mae, r2 = predict_and_evaluate(X_test, y_test, model)
    df_labels = load_labels(fd_num)

    # ---- Metrics (only RMSE, MAE, R²)
    metrics = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("RMSE", className="text-center"),
                html.H4(f"{rmse:.2f}", className="text-center text-primary")
            ])
        ], color="light"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("MAE", className="text-center"),
                html.H4(f"{mae:.2f}", className="text-center text-primary")
            ])
        ], color="light"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("R² Score", className="text-center"),
                html.H4(f"{r2:.3f}", className="text-center text-primary")
            ])
        ], color="light"), width=3),
    ], className="mb-3 justify-content-center")

    # ---- Scatter Plot
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(color='blue', size=4, opacity=0.6),
        name='Predictions'
    ))
    scatter_fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Ideal'
    ))
    scatter_fig.update_layout(
        title="Predicted vs Actual RUL",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
        template="plotly_white"
    )

    # ---- Residual Histogram
    residuals = y_test - y_pred
    residual_fig = go.Figure()
    residual_fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, marker_color='orange', opacity=0.7
    ))
    residual_fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Count",
        template="plotly_white"
    )

    # ---- RUL Trend Plot (first 100 samples)
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(y=y_test[:100], mode='lines', name='Actual'))
    trend_fig.add_trace(go.Scatter(y=y_pred[:100], mode='lines', name='Predicted'))
    trend_fig.update_layout(
        title="RUL Trend (First 100 Samples)",
        xaxis_title="Sample Index",
        yaxis_title="RUL",
        template="plotly_white"
    )

    # ---- Labels Table
    if not df_labels.empty:
        table = dbc.Table.from_dataframe(df_labels.head(10), striped=True, bordered=True, hover=True, size='sm')
    else:
        table = html.Div("No label data available.")

    return metrics, scatter_fig, residual_fig, trend_fig, table

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    print("Starting RUL Prediction Dashboard...")
    print("Open your browser and navigate to: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
