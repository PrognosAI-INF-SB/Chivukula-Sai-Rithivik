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


def predict_and_evaluate(X_test, y_test, model):
    """Predict RUL and compute evaluation metrics."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    return y_pred, rmse, mae, r2, residuals

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
                        className="text-center mt-4 mb-4"))
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Dataset"),
            dcc.Dropdown(
                id='fd-selector',
                options=[{'label': f'FD00{i}', 'value': i} for i in FD_DATASETS],
                value=4,
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
        dbc.Col([
            html.H5("Sample Predictions"),
            html.Div(id='predictions-table')
        ])
    ], className="mb-4")
], fluid=True)

# =============================================================================
# CALLBACK
# =============================================================================
@app.callback(
    [Output('metrics-display', 'children'),
     Output('scatter-plot', 'figure'),
     Output('residual-hist', 'figure'),
     Output('predictions-table', 'children')],
    [Input('fd-selector', 'value')]
)
def update_dashboard(fd_num):
    X_test, y_test, model = load_data_and_model(fd_num)
    if X_test is None:
        return "Error loading data.", go.Figure(), go.Figure(), "No data available."

    y_pred, rmse, mae, r2, residuals = predict_and_evaluate(X_test, y_test, model)

    # ---- Metrics (neatly spaced)
    metrics = dbc.Row([
        dbc.Col(html.Div([
            html.H5("RMSE", style={'text-align': 'center'}),
            html.P(f"{rmse:.2f}", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'})
        ]), width=2),

        dbc.Col(html.Div([
            html.H5("MAE", style={'text-align': 'center'}),
            html.P(f"{mae:.2f}", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'})
        ]), width=2),

        dbc.Col(html.Div([
            html.H5("R² Score", style={'text-align': 'center'}),
            html.P(f"{r2:.3f}", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'})
        ]), width=2),

        dbc.Col(html.Div([
            html.H5("Test Samples", style={'text-align': 'center'}),
            html.P(f"{len(y_test)}", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '18px'})
        ]), width=2),
    ], className="mb-2")

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

    # ---- Table (first 10 samples)
    df = pd.DataFrame({
        "Sample": np.arange(10),
        "Actual RUL": y_test[:10].round(2),
        "Predicted RUL": y_pred[:10].round(2),
        "Residual": residuals[:10].round(2)
    })
    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size='sm')

    return metrics, scatter_fig, residual_fig, table

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    print("Starting RUL Prediction Dashboard...")
    print("Open your browser and navigate to: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
