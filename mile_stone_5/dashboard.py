import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
import base64
import io
import os

# =============================================================================
# CONFIGURATIONS
# =============================================================================
sequences_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone1 results"
models_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone2 results"
data_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/"
sequence_length = 30
WARNING_THRESHOLD = 50
CRITICAL_THRESHOLD = 20
SAMPLE_LIMIT = 100

# =============================================================================
# CUSTOM CSS
# =============================================================================
custom_css = """
.card-header-custom {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px 10px 0 0;
    font-weight: bold;
}

.info-card {
    border-left: 4px solid #667eea;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}

.metric-card {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.upload-zone {
    border: 3px dashed #667eea;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    transition: all 0.3s;
}

.upload-zone:hover {
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    border-color: #764ba2;
}

.data-guide-table {
    font-size: 0.9em;
}

.section-divider {
    border-top: 2px solid #667eea;
    margin: 30px 0;
}
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_sequences(fd):
    """Load pre-saved sequences from NPZ file"""
    npz_path = os.path.join(sequences_dir, f"FD00{fd}_sequences.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["X_test"], data["y_test"]
    return None, None

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return predictions and metrics"""
    y_pred = model.predict(X_test).flatten()
    y_pred = np.maximum(y_pred, 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, r2

def label_rul_status(y_pred):
    """Label RUL predictions as CRITICAL, WARNING, or NORMAL"""
    labels = []
    for r in y_pred:
        if r <= CRITICAL_THRESHOLD:
            labels.append("CRITICAL")
        elif r <= WARNING_THRESHOLD:
            labels.append("WARNING")
        else:
            labels.append("NORMAL")
    return labels

def preprocess_uploaded_data(df, scaler, feature_cols):
    """Preprocess uploaded CSV data"""
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    for sensor in drop_sensors:
        if sensor in df.columns:
            df.drop(sensor, axis=1, inplace=True)
    
    sensor_features = [col for col in df.columns if col.startswith('s_')]
    
    for sensor in sensor_features:
        df[f'{sensor}_roll_mean'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df[f'{sensor}_roll_std'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df

def create_test_sequences(df, feature_cols):
    """Create sequences from preprocessed data"""
    X_test = []
    unit_ids = []
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id][feature_cols].values
        
        if len(unit_data) >= sequence_length:
            X_test.append(unit_data[-sequence_length:])
        else:
            pad_length = sequence_length - len(unit_data)
            padded = np.vstack([np.zeros((pad_length, unit_data.shape[1])), unit_data])
            X_test.append(padded)
        
        unit_ids.append(unit_id)
    
    return np.array(X_test), unit_ids

def load_scaler_and_features(fd):
    """Load scaler and feature columns for a given FD dataset"""
    index_cols = ['unit_id', 'time_cycles']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    col_names = index_cols + setting_cols + sensor_cols
    
    train = pd.read_csv(os.path.join(data_dir, f"train_and_test/train_FD00{fd}.txt"), 
                        sep=' ', header=None)
    train.dropna(axis=1, inplace=True)
    train.columns = col_names
    
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    train.drop(drop_sensors, axis=1, inplace=True)
    
    sensor_features = [col for col in train.columns if col.startswith('s_')]
    
    for sensor in sensor_features:
        train[f'{sensor}_roll_mean'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        train[f'{sensor}_roll_std'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
    
    feature_cols = [col for col in train.columns if col not in ['unit_id', 'time_cycles']]
    
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    
    return scaler, feature_cols

def normalize_column_names(df):
    """Normalize column names to match expected format"""
    column_mapping = {
        'unit_number': 'unit_id',
        'time_in_cycles': 'time_cycles',
        'operational_setting_1': 'setting_1',
        'operational_setting_2': 'setting_2',
        'operational_setting_3': 'setting_3',
    }
    
    for i in range(1, 22):
        column_mapping[f'sensor_{i}'] = f's_{i}'
    
    df = df.rename(columns=column_mapping)
    
    return df

# =============================================================================
# INITIALIZE DASH APP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.LUX,
    'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css'
])
app.title = "NASA CMAPSS RUL Prediction Dashboard"

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        ''' + custom_css + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# DATA FORMAT GUIDE COMPONENT
# =============================================================================
def create_data_guide():
    """Create comprehensive data format guide"""
    
    # Sample data structure
    sample_data = pd.DataFrame({
        'unit_id': [1, 1, 1, 2, 2],
        'time_cycles': [1, 2, 3, 1, 2],
        'setting_1': [0.0023, 0.0027, 0.0003, 0.0042, 0.0014],
        'setting_2': [0.0003, 0.0003, 0.0001, 0.0000, 0.0000],
        'setting_3': [100.0, 100.0, 100.0, 100.0, 100.0],
        's_1': [518.67, 518.67, 518.67, 518.67, 518.67],
        's_2': [641.82, 642.15, 642.35, 642.44, 642.32],
        '...': ['...', '...', '...', '...', '...']
    })
    
    guide = dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="bi bi-info-circle-fill me-2"),
                "📋 Data Format Guide"
            ], className="mb-0")
        ], className="card-header-custom"),
        dbc.CardBody([
            # Overview
            dbc.Alert([
                html.H5("📊 Expected Data Structure", className="alert-heading"),
                html.P([
                    "Your CSV file must contain time-series data for turbofan engine units. ",
                    "Each row represents one time cycle for a specific engine unit."
                ])
            ], color="info", className="mb-4"),
            
            # Required Columns
            html.H5("✅ Required Columns", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Identification Columns", className="text-primary"),
                            html.Ul([
                                html.Li([html.Code("unit_id"), " or ", html.Code("unit_number"), 
                                        " — Unique engine identifier"]),
                                html.Li([html.Code("time_cycles"), " or ", html.Code("time_in_cycles"), 
                                        " — Time step for the unit"])
                            ])
                        ])
                    ], className="mb-3"),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Operational Settings", className="text-success"),
                            html.Ul([
                                html.Li([html.Code("setting_1"), " or ", html.Code("operational_setting_1")]),
                                html.Li([html.Code("setting_2"), " or ", html.Code("operational_setting_2")]),
                                html.Li([html.Code("setting_3"), " or ", html.Code("operational_setting_3")])
                            ])
                        ])
                    ], className="mb-3"),
                ], width=6),
            ]),
            
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sensor Measurements (21 sensors)", className="text-danger"),
                    html.P([
                        html.Code("s_1"), " to ", html.Code("s_21"), 
                        " OR ", 
                        html.Code("sensor_1"), " to ", html.Code("sensor_21")
                    ], className="mb-2"),
                    html.Small("Each sensor column contains measurement values at each time cycle.", 
                              className="text-muted")
                ])
            ], className="mb-4"),
            
            # Sample Data Preview
            html.H5("📄 Sample Data Structure", className="mb-3"),
            dash_table.DataTable(
                data=sample_data.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sample_data.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 
                    'padding': '10px',
                    'fontSize': '11px',
                    'fontFamily': 'monospace'
                },
                style_header={
                    'backgroundColor': '#667eea',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': ['unit_id', 'time_cycles']},
                        'backgroundColor': '#e3f2fd'
                    }
                ]
            ),
            
            html.Hr(className="my-4"),
            
            # Important Notes
            dbc.Alert([
                html.H6("⚠️ Important Notes:", className="alert-heading"),
                html.Ul([
                    html.Li("Data should be in CSV format with headers"),
                    html.Li("All sensor columns must be numeric"),
                    html.Li("Missing values should be handled before upload"),
                ], className="mb-0")
            ], color="warning")
        ])
    ], className="mb-4 shadow")
    
    return guide

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = html.Div([
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        html.I(className="bi bi-cpu-fill me-3"),
                        "NASA CMAPSS RUL Prediction Dashboard"
                    ], className="text-center mb-2",
                       style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                              'WebkitBackgroundClip': 'text',
                              'WebkitTextFillColor': 'transparent',
                              'fontWeight': 'bold'}),
                    html.P("Remaining Useful Life Prediction for Turbofan Engines", 
                          className="text-center text-muted")
                ])
            ])
        ], className="mb-4"),
        
        dbc.Tabs([
            # =====================================================================
            # TAB 1: EXISTING DATASETS
            # =====================================================================
            dbc.Tab(label="📊 Pre-trained Datasets", tab_style={"marginLeft": "auto"}, children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("Analyze Pre-trained Models (FD001-FD004)", className="mb-0")
                                ], className="card-header-custom"),
                                dbc.CardBody([
                                    html.P(f"Select a dataset to view evaluation results for the first {SAMPLE_LIMIT} samples."),
                                    
                                    html.Label("Select Dataset:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='dataset-dropdown',
                                        options=[
                                            {'label': '🔧 FD001 - Single Operating Condition', 'value': '1'},
                                            {'label': '🔧 FD002 - Multiple Operating Conditions', 'value': '2'},
                                            {'label': '🔧 FD003 - Single Operating Condition', 'value': '3'},
                                            {'label': '🔧 FD004 - Multiple Operating Conditions', 'value': '4'}
                                        ],
                                        value='1',
                                        clearable=False,
                                        className="mb-3"
                                    ),
                                    
                                    dbc.Button("Load and Evaluate", id='load-button', 
                                               color="primary", size="lg", className="w-100")
                                ])
                            ], className="shadow mb-4")
                        ])
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner(html.Div(id='loading-output'), color="primary")
                        ])
                    ]),
                    
                    # Metrics Cards
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.I(className="bi bi-graph-up", 
                                              style={'fontSize': '2em', 'color': '#667eea'}),
                                        html.H5("RMSE", className="card-title mt-2"),
                                        html.H2(id='rmse-metric', className="text-primary")
                                    ], className="text-center")
                                ])
                            ], className="metric-card")
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.I(className="bi bi-check-circle", 
                                              style={'fontSize': '2em', 'color': '#28a745'}),
                                        html.H5("MAE", className="card-title mt-2"),
                                        html.H2(id='mae-metric', className="text-success")
                                    ], className="text-center")
                                ])
                            ], className="metric-card")
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.I(className="bi bi-star-fill", 
                                              style={'fontSize': '2em', 'color': '#17a2b8'}),
                                        html.H5("R² Score", className="card-title mt-2"),
                                        html.H2(id='r2-metric', className="text-info")
                                    ], className="text-center")
                                ])
                            ], className="metric-card")
                        ], width=4)
                    ], className="mb-4", id='metrics-row', style={'display': 'none'}),
                    
                    # Plots
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='scatter-plot')
                        ], width=6),
                        
                        dbc.Col([
                            dcc.Graph(id='residual-hist')
                        ], width=6)
                    ], className="mb-4", id='plots-row', style={'display': 'none'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='residual-trend')
                        ], width=12)
                    ], className="mb-4", id='trend-row', style={'display': 'none'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='condition-pie-chart')
                        ], width=12)
                    ], className="mb-4", id='pie-row', style={'display': 'none'}),

                    # Data Table
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📋 Prediction Results Table", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id='results-table',
                                        page_size=15,
                                        style_table={'overflowX': 'auto'},
                                        style_cell={'textAlign': 'left', 'padding': '10px'},
                                        style_header={'backgroundColor': '#667eea', 
                                                     'color': 'white',
                                                     'fontWeight': 'bold'},
                                        style_data_conditional=[
                                            {'if': {'filter_query': '{Condition_Label} = "CRITICAL"'},
                                             'backgroundColor': '#ffcccc', 'color': 'darkred', 'fontWeight': 'bold'},
                                            {'if': {'filter_query': '{Condition_Label} = "WARNING"'},
                                             'backgroundColor': '#fff4cc', 'color': 'darkorange', 'fontWeight': 'bold'},
                                            {'if': {'filter_query': '{Condition_Label} = "NORMAL"'},
                                             'backgroundColor': '#ccffcc', 'color': 'darkgreen'}
                                        ]
                                    )
                                ])
                            ], className="shadow")
                        ])
                    ], className="mb-4", id='table-row', style={'display': 'none'})
                    
                ], fluid=True)
            ]),
            
            # =====================================================================
            # TAB 2: NEW DATASET UPLOAD
            # =====================================================================
            dbc.Tab(label="📤 Upload New Dataset", children=[
                dbc.Container([
                    # Data Format Guide
                    dbc.Row([
                        dbc.Col([
                            create_data_guide()
                        ])
                    ]),
                    
                    html.Hr(className="section-divider"),
                    
                    # Upload Section
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("🚀 Upload and Predict RUL", className="mb-0")
                                ], className="card-header-custom"),
                                dbc.CardBody([
                                    html.Label("Select Model for Prediction:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='model-dropdown',
                                        options=[
                                            {'label': 'FD001 Model', 'value': '1'},
                                            {'label': 'FD002 Model', 'value': '2'},
                                            {'label': 'FD003 Model', 'value': '3'},
                                            {'label': 'FD004 Model', 'value': '4'}
                                        ],
                                        value='1',
                                        clearable=False,
                                        className="mb-4"
                                    ),
                                    
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            html.I(className="bi bi-cloud-upload", 
                                                  style={'fontSize': '3em', 'color': '#667eea'}),
                                            html.H5('Drag and Drop or Click to Select CSV File', 
                                                   className="mt-3 mb-2"),
                                            html.P("Maximum file size: 200MB", 
                                                  className="text-muted small")
                                        ], className="upload-zone"),
                                        multiple=False
                                    )
                                ])
                            ], className="shadow mb-4")
                        ])
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='upload-status', className="mb-3")
                        ])
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className="bi bi-play-fill me-2"),
                                "Process and Predict"
                            ], id='predict-button', 
                                       color="success", size="lg", className="w-100",
                                       disabled=True)
                        ])
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner(html.Div(id='upload-loading'), color="success")
                        ])
                    ]),
                    
                    # Upload Results
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("👁️ Preview of Uploaded Data", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='upload-preview')
                                ])
                            ], className="shadow")
                        ])
                    ], className="mb-4", id='preview-row', style={'display': 'none'}),
                    
                    # Prediction Results Graphs
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='upload-scatter')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='upload-histogram')
                        ], width=6)
                    ], className="mb-4", id='upload-plot-row', style={'display': 'none'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='upload-pie-chart')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='upload-box-plot')
                        ], width=6)
                    ], className="mb-4", id='upload-plot-row2', style={'display': 'none'}),
                    
                    # Prediction Results Table
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Prediction Results", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        id='upload-results-table',
                                        page_size=15,
                                        style_table={'overflowX': 'auto'},
                                        style_cell={'textAlign': 'left', 'padding': '10px'},
                                        style_header={'backgroundColor': '#28a745',
                                                     'color': 'white',
                                                     'fontWeight': 'bold'},
                                        style_data_conditional=[
                                            {'if': {'filter_query': '{Status} = "CRITICAL"'},
                                             'backgroundColor': '#ffcccc', 'color': 'darkred', 'fontWeight': 'bold'},
                                            {'if': {'filter_query': '{Status} = "WARNING"'},
                                             'backgroundColor': '#fff4cc', 'color': 'darkorange', 'fontWeight': 'bold'},
                                            {'if': {'filter_query': '{Status} = "NORMAL"'},
                                             'backgroundColor': '#ccffcc', 'color': 'darkgreen'}
                                        ],
                                        export_format='csv',
                                        export_headers='display'
                                    )
                                ])
                            ], className="shadow")
                        ])
                    ], className="mb-4", id='upload-table-row', style={'display': 'none'})
                    
                ], fluid=True)
            ])
        ])
    ], fluid=True, className="p-4")
], style={'backgroundColor': '#f5f7fa', 'minHeight': '100vh'})

# =============================================================================
# CALLBACKS - TAB 1: EXISTING DATASETS
# =============================================================================
@app.callback(
    [Output('rmse-metric', 'children'),
     Output('mae-metric', 'children'),
     Output('r2-metric', 'children'),
     Output('scatter-plot', 'figure'),
     Output('residual-hist', 'figure'),
     Output('residual-trend', 'figure'),
     Output('condition-pie-chart', 'figure'),
     Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('metrics-row', 'style'),
     Output('plots-row', 'style'),
     Output('trend-row', 'style'),
     Output('pie-row', 'style'),
     Output('table-row', 'style'),
     Output('loading-output', 'children')],
    [Input('load-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_existing_dataset(n_clicks, fd):
    sample_limit = SAMPLE_LIMIT

    if n_clicks is None:
        return ["", "", "", {}, {}, {}, {}, [], [], 
                {'display': 'none'}, {'display': 'none'}, 
                {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, ""]
    
    try:
        model_path = os.path.join(models_dir, f"fd00{fd}_model.h5")
        if not os.path.exists(model_path):
            return ["Error", "Error", "Error", 
                    {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 
                    f"Error: Model not found at {model_path}"]
        
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        
        X_test, y_test = load_sequences(fd)
        if X_test is None:
            return ["Error", "Error", "Error", 
                    {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 
                    f"Error: Sequences not found for FD00{fd}"]
        
        y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        labels = label_rul_status(y_pred)
        
        y_test_limited = y_test[:sample_limit]
        y_pred_limited = y_pred[:sample_limit]
        residuals = y_test_limited - y_pred_limited
        
        rmse_text = f"{rmse:.4f}"
        mae_text = f"{mae:.4f}"
        r2_text = f"{r2:.4f}"
        
        # 1. Scatter plot
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=y_test_limited, y=y_pred_limited,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6, color='#667eea',
                       line=dict(width=1, color='white'))
        ))
        scatter_fig.add_trace(go.Scatter(
            x=[min(y_test_limited), max(y_test_limited)],
            y=[min(y_test_limited), max(y_test_limited)],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash', width=2)
        ))
        scatter_fig.update_layout(
            title=f"Predicted vs Actual RUL (FD00{fd}) - First {sample_limit} Samples",
            xaxis_title="Actual RUL",
            yaxis_title="Predicted RUL",
            height=400,
            template='plotly_white',
            hovermode='closest'
        )
        
        # 2. Residual Histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=residuals, 
            nbinsx=40, 
            name='Residuals',
            marker=dict(color='#764ba2', opacity=0.7,
                       line=dict(color='white', width=1))
        ))
        hist_fig.update_layout(
            title=f"Residual Distribution (FD00{fd})",
            xaxis_title="Residual (Actual - Predicted)",
            yaxis_title="Frequency",
            height=400,
            template='plotly_white'
        )
        
        # 3. Residual Trend Plot
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ))
        trend_fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        trend_fig.update_layout(
            title=f"Residual Trend (First {sample_limit} Samples) — FD00{fd}",
            xaxis_title="Sample Index",
            yaxis_title="Residual",
            height=400,
            template='plotly_white'
        )
        
        # 4. Condition Label Pie Chart
        status_counts = pd.Series(labels[:sample_limit]).value_counts().reset_index()
        status_counts.columns = ['Condition', 'Count']
        
        pie_fig = px.pie(
            status_counts, 
            values='Count', 
            names='Condition', 
            title=f'RUL Condition Distribution (FD00{fd}) - First {sample_limit} Samples',
            color='Condition',
            color_discrete_map={
                'CRITICAL': '#dc3545', 
                'WARNING': '#ffc107', 
                'NORMAL': '#28a745'
            },
            height=400,
            hole=0.4
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(template='plotly_white')
        
        # 5. Table
        table_df = pd.DataFrame({
            "Sample_Index": np.arange(len(y_test))[:sample_limit],
            "Actual_RUL": np.round(y_test[:sample_limit], 2),
            "Predicted_RUL": np.round(y_pred[:sample_limit], 2),
            "Condition_Label": labels[:sample_limit]
        })
        
        table_data = table_df.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in table_df.columns]
        
        return [rmse_text, mae_text, r2_text, 
                scatter_fig, hist_fig, trend_fig, pie_fig,
                table_data, table_columns, 
                {'display': 'flex'}, {'display': 'flex'}, 
                {'display': 'flex'}, {'display': 'flex'},
                {'display': 'block'}, 
                "✓ Analysis Complete"]
        
    except Exception as e:
        return ["Error", "Error", "Error", 
                {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 
                f"Error: {str(e)}"]

# =============================================================================
# CALLBACKS - TAB 2: UPLOAD NEW DATASET
# =============================================================================
@app.callback(
    [Output('upload-status', 'children'),
     Output('predict-button', 'disabled'),
     Output('upload-preview', 'children'),
     Output('preview-row', 'style')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def handle_upload(contents, filename):
    if contents is None:
        return ["", True, "", {'display': 'none'}]
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        df = normalize_column_names(df)
        
        required_cols = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                        [f's_{i}' for i in range(1, 22)]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            status = dbc.Alert([
                html.H5("❌ Missing Required Columns", className="alert-heading"),
                html.P("Your file is missing the following columns:"),
                html.P(html.Code(', '.join(missing_cols)), className="mb-2"),
                html.Hr(),
                html.P([
                    "Please refer to the Data Format Guide above for the expected column structure."
                ], className="mb-0 small")
            ], color="danger")
            return [status, True, "", {'display': 'none'}]
        
        status = dbc.Alert([
            html.H5("✅ File Uploaded Successfully!", className="alert-heading"),
            html.P([
                html.Strong(filename), 
                f" — {len(df):,} rows × {len(df.columns)} columns"
            ]),
            html.P([
                html.Strong(f"{len(df['unit_id'].unique())} unique engine units detected")
            ], className="mb-0")
        ], color="success")
        
        preview = dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
            style_header={
                'backgroundColor': '#667eea',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': ['unit_id', 'time_cycles']},
                    'backgroundColor': '#e3f2fd',
                    'fontWeight': 'bold'
                }
            ]
        )
        
        return [status, False, preview, {'display': 'block'}]
        
    except Exception as e:
        status = dbc.Alert([
            html.H5("❌ Error Reading File", className="alert-heading"),
            html.P(str(e))
        ], color="danger")
        return [status, True, "", {'display': 'none'}]

@app.callback(
    [Output('upload-results-table', 'data'),
     Output('upload-results-table', 'columns'),
     Output('upload-scatter', 'figure'),
     Output('upload-histogram', 'figure'),
     Output('upload-pie-chart', 'figure'),
     Output('upload-box-plot', 'figure'),
     Output('upload-table-row', 'style'),
     Output('upload-plot-row', 'style'),
     Output('upload-plot-row2', 'style'),
     Output('upload-loading', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('model-dropdown', 'value')]
)
def predict_uploaded_data(n_clicks, contents, model_fd):
    if n_clicks is None or contents is None:
        return [[], [], {}, {}, {}, {}, 
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, ""]
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        df = normalize_column_names(df)
        
        model_path = os.path.join(models_dir, f"fd00{model_fd}_model.h5")
        if not os.path.exists(model_path):
            return [[], [], {}, {}, {}, {}, 
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                    f"Error: Model not found at {model_path}"]
        
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        
        scaler, feature_cols = load_scaler_and_features(model_fd)
        
        df_processed = preprocess_uploaded_data(df.copy(), scaler, feature_cols)
        
        X_test, unit_ids = create_test_sequences(df_processed, feature_cols)
        
        y_pred = model.predict(X_test).flatten()
        y_pred = np.maximum(y_pred, 0)
        
        labels = label_rul_status(y_pred)
        
        # Create results table
        results_df = pd.DataFrame({
            "Unit_ID": unit_ids,
            "Predicted_RUL": np.round(y_pred, 2),
            "Status": labels
        })
        
        table_data = results_df.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in results_df.columns]
        
        # 1. Scatter Plot - RUL by Unit
        scatter_fig = go.Figure()
        
        colors = {'CRITICAL': '#dc3545', 'WARNING': '#ffc107', 'NORMAL': '#28a745'}
        for status in ['NORMAL', 'WARNING', 'CRITICAL']:
            mask = results_df['Status'] == status
            scatter_fig.add_trace(go.Scatter(
                x=results_df[mask]['Unit_ID'],
                y=results_df[mask]['Predicted_RUL'],
                mode='markers',
                name=status,
                marker=dict(size=10, color=colors[status], 
                           line=dict(width=1, color='white'))
            ))
        
        scatter_fig.update_layout(
            title="Predicted RUL by Unit ID",
            xaxis_title="Unit ID",
            yaxis_title="Predicted RUL (cycles)",
            height=400,
            template='plotly_white',
            hovermode='closest'
        )
        
        # 2. Histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=y_pred, 
            nbinsx=25, 
            name='Predicted RUL',
            marker=dict(color='#667eea', opacity=0.7,
                       line=dict(color='white', width=1))
        ))
        hist_fig.update_layout(
            title="Distribution of Predicted RUL",
            xaxis_title="Predicted RUL (cycles)",
            yaxis_title="Frequency",
            height=400,
            template='plotly_white'
        )
        
        # 3. Pie Chart - Status Distribution
        status_counts = results_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        pie_fig = px.pie(
            status_counts,
            values='Count',
            names='Status',
            title='Engine Health Status Distribution',
            color='Status',
            color_discrete_map={
                'CRITICAL': '#dc3545',
                'WARNING': '#ffc107',
                'NORMAL': '#28a745'
            },
            height=400,
            hole=0.4
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(template='plotly_white')
        
        # 4. Box Plot - RUL Distribution by Status
        box_fig = go.Figure()
        
        for status in ['CRITICAL', 'WARNING', 'NORMAL']:
            mask = results_df['Status'] == status
            box_fig.add_trace(go.Box(
                y=results_df[mask]['Predicted_RUL'],
                name=status,
                marker=dict(color=colors[status]),
                boxmean='sd'
            ))
        
        box_fig.update_layout(
            title="RUL Distribution by Health Status",
            yaxis_title="Predicted RUL (cycles)",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return [table_data, table_columns, 
                scatter_fig, hist_fig, pie_fig, box_fig,
                {'display': 'block'}, {'display': 'flex'}, {'display': 'flex'},
                "✓ Prediction Complete"]
        
    except Exception as e:
        return [[], [], {}, {}, {}, {},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                f"Error: {str(e)}"]

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)