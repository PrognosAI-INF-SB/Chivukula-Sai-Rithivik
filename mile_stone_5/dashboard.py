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
# NOTE: The directories below are placeholders and must be valid paths 
# in the actual execution environment.
sequences_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone1 results"
models_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/mile stone2 results"
data_dir = "C:/Users/chsai/OneDrive/Desktop/infosys internship/dataset/CMaps/"
sequence_length = 30
WARNING_THRESHOLD = 50
CRITICAL_THRESHOLD = 20
# Hardcode Sample Limit to 100 as requested
SAMPLE_LIMIT = 100

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

# Removed calculate_bias_metrics function

def preprocess_uploaded_data(df, scaler, feature_cols):
    """Preprocess uploaded CSV data"""
    # Drop constant sensors
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    for sensor in drop_sensors:
        if sensor in df.columns:
            df.drop(sensor, axis=1, inplace=True)
    
    # Get sensor features
    sensor_features = [col for col in df.columns if col.startswith('s_')]
    
    # Add rolling features
    for sensor in sensor_features:
        df[f'{sensor}_roll_mean'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df[f'{sensor}_roll_std'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
    
    # Scale features
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
    # Load training data to fit scaler
    index_cols = ['unit_id', 'time_cycles']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    col_names = index_cols + setting_cols + sensor_cols
    
    train = pd.read_csv(os.path.join(data_dir, f"train_and_test/train_FD00{fd}.txt"), 
                        sep=' ', header=None)
    train.dropna(axis=1, inplace=True)
    train.columns = col_names
    
    # Drop constant sensors
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    train.drop(drop_sensors, axis=1, inplace=True)
    
    # Get sensor features
    sensor_features = [col for col in train.columns if col.startswith('s_')]
    
    # Add rolling features
    for sensor in sensor_features:
        train[f'{sensor}_roll_mean'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        train[f'{sensor}_roll_std'] = train.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
    
    # Get feature columns
    feature_cols = [col for col in train.columns if col not in ['unit_id', 'time_cycles']]
    
    # Fit scaler
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    
    return scaler, feature_cols

# =============================================================================
# INITIALIZE DASH APP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NASA CMAPSS RUL Prediction Dashboard"

# =============================================================================
# LAYOUT
# =============================================================================
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("NASA CMAPSS RUL Prediction Dashboard", 
                    className="text-center text-primary mb-4"),
            # Removed the <html.P> descriptive line
        ])
    ]),
    
    dbc.Tabs([
        # =====================================================================
        # TAB 1: EXISTING DATASETS
        # =====================================================================
        dbc.Tab(label="Pre-trained Datasets", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("Analyze Pre-trained Models (FD001-FD004)", className="mt-4 mb-3"),
                        html.P(f"Select a dataset to view evaluation results for the first **{SAMPLE_LIMIT}** samples."),
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[
                                {'label': 'FD001', 'value': '1'},
                                {'label': 'FD002', 'value': '2'},
                                {'label': 'FD003', 'value': '3'},
                                {'label': 'FD004', 'value': '4'}
                            ],
                            value='1',
                            clearable=False
                        )
                    ], width=12),
                    
                    # Removed Sample Limit Slider
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Load and Evaluate", id='load-button', 
                                   color="primary", size="lg", className="w-100")
                    ], width=12)
                ], className="mb-4"),
                
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
                                html.H5("RMSE", className="card-title"),
                                html.H2(id='rmse-metric', className="text-primary")
                            ])
                        ])
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("MAE", className="card-title"),
                                html.H2(id='mae-metric', className="text-success")
                            ])
                        ])
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("R² Score", className="card-title"),
                                html.H2(id='r2-metric', className="text-info")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4", id='metrics-row', style={'display': 'none'}),
                
                # Plots (Scatter and Residual Histogram)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='scatter-plot')
                    ], width=6),
                    
                    dbc.Col([
                        dcc.Graph(id='residual-hist') # Restored
                    ], width=6)
                ], className="mb-4", id='plots-row', style={'display': 'none'}),
                
                # Residual Trend Plot (Restored)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='residual-trend') 
                    ], width=12)
                ], className="mb-4", id='trend-row', style={'display': 'none'}),
                
                # RUL Condition Pie Chart (New)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='condition-pie-chart')
                    ], width=12)
                ], className="mb-4", id='pie-row', style={'display': 'none'}),

                # Data Table
                dbc.Row([
                    dbc.Col([
                        html.H5("Prediction Results Table", className="mb-3"),
                        dash_table.DataTable(
                            id='results-table',
                            page_size=15,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{Condition_Label} = "CRITICAL"'},
                                 'backgroundColor': '#ffcccc', 'color': 'darkred'},
                                {'if': {'filter_query': '{Condition_Label} = "WARNING"'},
                                 'backgroundColor': '#fff4cc', 'color': 'darkorange'},
                                {'if': {'filter_query': '{Condition_Label} = "NORMAL"'},
                                 'backgroundColor': '#ccffcc', 'color': 'darkgreen'}
                            ]
                        )
                    ])
                ], className="mb-4", id='table-row', style={'display': 'none'})
                
            ], fluid=True)
        ]),
        
        # =====================================================================
        # TAB 2: NEW DATASET UPLOAD
        # =====================================================================
        dbc.Tab(label="Upload New Dataset", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("Upload and Predict RUL for New Dataset", className="mt-4 mb-3"),
                        html.P("Upload a CSV file with the required columns and select a model for prediction."),
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Model for Prediction:", className="fw-bold"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': 'FD001 Model', 'value': '1'},
                                {'label': 'FD002 Model', 'value': '2'},
                                {'label': 'FD003 Model', 'value': '3'},
                                {'label': 'FD004 Model', 'value': '4'}
                            ],
                            value='1',
                            clearable=False
                        )
                    ], width=6)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select CSV File', style={'color': 'blue', 'cursor': 'pointer'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '80px',
                                'lineHeight': '80px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'backgroundColor': '#f8f9fa'
                            },
                            multiple=False
                        )
                    ])
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id='upload-status', className="mb-3")
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Process and Predict", id='predict-button', 
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
                        html.H5("Preview of Uploaded Data", className="mb-3"),
                        html.Div(id='upload-preview')
                    ])
                ], className="mb-4", id='preview-row', style={'display': 'none'}),
                
                # Prediction Results for Uploaded Data
                dbc.Row([
                    dbc.Col([
                        html.H5("Prediction Results", className="mb-3"),
                        dash_table.DataTable(
                            id='upload-results-table',
                            page_size=15,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{Status} = "CRITICAL"'},
                                 'backgroundColor': '#ffcccc', 'color': 'darkred'},
                                {'if': {'filter_query': '{Status} = "WARNING"'},
                                 'backgroundColor': '#fff4cc', 'color': 'darkorange'},
                                {'if': {'filter_query': '{Status} = "NORMAL"'},
                                 'backgroundColor': '#ccffcc', 'color': 'darkgreen'}
                            ]
                        )
                    ])
                ], className="mb-4", id='upload-table-row', style={'display': 'none'}),
                
                # Distribution Plot for Uploaded Data
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='upload-histogram')
                    ])
                ], className="mb-4", id='upload-plot-row', style={'display': 'none'})
                
            ], fluid=True)
        ])
    ])
], fluid=True, className="p-4")

# =============================================================================
# CALLBACKS - TAB 1: EXISTING DATASETS
# =============================================================================
@app.callback(
    [Output('rmse-metric', 'children'),
     Output('mae-metric', 'children'),
     Output('r2-metric', 'children'),
     Output('scatter-plot', 'figure'),
     Output('residual-hist', 'figure'),    # Restored
     Output('residual-trend', 'figure'),   # Restored
     Output('condition-pie-chart', 'figure'), # New
     Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('metrics-row', 'style'),
     Output('plots-row', 'style'),
     Output('trend-row', 'style'),         # Restored
     Output('pie-row', 'style'),           # New
     Output('table-row', 'style'),
     Output('loading-output', 'children')],
    [Input('load-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')]
)
def update_existing_dataset(n_clicks, fd):
    # Hardcoded sample limit
    sample_limit = SAMPLE_LIMIT

    # Initial state (15 return values)
    if n_clicks is None:
        return ["", "", "", {}, {}, {}, {}, [], [], 
                {'display': 'none'}, {'display': 'none'}, 
                {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, ""]
    
    try:
        # Load model
        model_path = os.path.join(models_dir, f"fd00{fd}_model.h5")
        if not os.path.exists(model_path):
            return ["Error", "Error", "Error", 
                    {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, f"Error: Model not found at {model_path}"]
        
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        
        # Load sequences
        X_test, y_test = load_sequences(fd)
        if X_test is None:
            return ["Error", "Error", "Error", 
                    {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, f"Error: Sequences not found for FD00{fd}"]
        
        # Evaluate
        y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        labels = label_rul_status(y_pred)
        
        # Limit samples for visualization
        y_test_limited = y_test[:sample_limit]
        y_pred_limited = y_pred[:sample_limit]
        residuals = y_test_limited - y_pred_limited # Restored for residual plots
        
        # Metrics
        rmse_text = f"{rmse:.4f}"
        mae_text = f"{mae:.4f}"
        r2_text = f"{r2:.4f}"
        
        # 1. Scatter plot
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=y_test_limited, y=y_pred_limited,
            mode='markers',
            name='Predictions',
            marker=dict(size=5, opacity=0.6)
        ))
        scatter_fig.add_trace(go.Scatter(
            x=[min(y_test_limited), max(y_test_limited)],
            y=[min(y_test_limited), max(y_test_limited)],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))
        scatter_fig.update_layout(
            title=f"Predicted vs Actual RUL (FD00{fd}) - First {sample_limit} Samples",
            xaxis_title="Actual RUL",
            yaxis_title="Predicted RUL",
            height=400
        )
        
        # 2. Residual Histogram (Restored)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=residuals, nbinsx=40, name='Residuals'))
        hist_fig.update_layout(
            title=f"Residual Distribution (FD00{fd})",
            xaxis_title="Residual (Actual - Predicted)",
            yaxis_title="Frequency",
            height=400
        )
        
        # 3. Residual Trend Plot (Restored)
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='blue')
        ))
        trend_fig.add_hline(y=0, line_dash="dash", line_color="red")
        trend_fig.update_layout(
            title=f"Residual Trend (First {sample_limit} Samples) — FD00{fd}",
            xaxis_title="Sample Index",
            yaxis_title="Residual",
            height=400
        )
        
        # 4. Condition Label Pie Chart (New)
        status_counts = pd.Series(labels[:sample_limit]).value_counts().reset_index()
        status_counts.columns = ['Condition', 'Count']
        
        pie_fig = px.pie(status_counts, values='Count', names='Condition', 
                         title=f'RUL Condition Label Distribution (FD00{fd}) - First {sample_limit} Samples',
                         color='Condition',
                         color_discrete_map={
                             'CRITICAL': 'Red', 
                             'WARNING': 'Orange', 
                             'NORMAL': 'Green'
                         },
                         height=400)
        
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
                scatter_fig, hist_fig, trend_fig, pie_fig, # Figures
                table_data, table_columns, 
                {'display': 'flex'}, {'display': 'flex'}, 
                {'display': 'flex'}, {'display': 'flex'},
                {'display': 'block'}, 
                "✓ Analysis Complete"]
        
    except Exception as e:
        return ["Error", "Error", "Error", 
                {}, {}, {}, {}, [], [], {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, f"Error: {str(e)}"]

# =============================================================================
# CALLBACKS - TAB 2: UPLOAD NEW DATASET
# =============================================================================
def normalize_column_names(df):
    """Normalize column names to match expected format"""
    # Create a mapping for different naming conventions
    column_mapping = {
        'unit_number': 'unit_id',
        'time_in_cycles': 'time_cycles',
        'operational_setting_1': 'setting_1',
        'operational_setting_2': 'setting_2',
        'operational_setting_3': 'setting_3',
    }
    
    # Add sensor mappings
    for i in range(1, 22):
        column_mapping[f'sensor_{i}'] = f's_{i}'
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    return df

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
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Check required columns
        required_cols = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                        [f's_{i}' for i in range(1, 22)]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            status = dbc.Alert([
                html.H5("❌ Missing Required Columns", className="alert-heading"),
                html.P("Your file is missing the following columns:"),
                html.P(f"{', '.join(missing_cols)}", className="mb-2"),
                html.Hr(),
                html.P([
                    "Expected column names: ",
                    html.Code("unit_id, time_cycles, setting_1, setting_2, setting_3, s_1, s_2, ..., s_21"),
                    html.Br(),
                    "OR alternate names: ",
                    html.Code("unit_number, time_in_cycles, operational_setting_1-3, sensor_1-21")
                ], className="mb-0 small")
            ], color="danger")
            return [status, True, "", {'display': 'none'}]
        
        # Success
        status = dbc.Alert(
            f"✓ File '{filename}' uploaded successfully! ({len(df)} rows, {len(df.columns)} columns)",
            color="success"
        )
        
        # Preview
        preview = dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
        
        return [status, False, preview, {'display': 'block'}]
        
    except Exception as e:
        status = dbc.Alert(f"❌ Error reading file: {str(e)}", color="danger")
        return [status, True, "", {'display': 'none'}]

@app.callback(
    [Output('upload-results-table', 'data'),
     Output('upload-results-table', 'columns'),
     Output('upload-histogram', 'figure'),
     Output('upload-table-row', 'style'),
     Output('upload-plot-row', 'style'),
     Output('upload-loading', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('model-dropdown', 'value')]
)
def predict_uploaded_data(n_clicks, contents, model_fd):
    if n_clicks is None or contents is None:
        return [[], [], {}, {'display': 'none'}, {'display': 'none'}, ""]
    
    try:
        # Parse uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Load model
        model_path = os.path.join(models_dir, f"fd00{model_fd}_model.h5")
        if not os.path.exists(model_path):
            return [[], [], {}, {'display': 'none'}, {'display': 'none'}, 
                    f"Error: Model not found at {model_path}"]
        
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        
        # Load scaler and features
        scaler, feature_cols = load_scaler_and_features(model_fd)
        
        # Preprocess
        df_processed = preprocess_uploaded_data(df.copy(), scaler, feature_cols)
        
        # Create sequences
        X_test, unit_ids = create_test_sequences(df_processed, feature_cols)
        
        # Predict
        y_pred = model.predict(X_test).flatten()
        y_pred = np.maximum(y_pred, 0)
        
        # Label status
        labels = label_rul_status(y_pred)
        
        # Create results table
        results_df = pd.DataFrame({
            "Unit_ID": unit_ids,
            "Predicted_RUL": np.round(y_pred, 2),
            "Status": labels
        })
        
        table_data = results_df.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in results_df.columns]
        
        # Create histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=y_pred, nbinsx=25, name='Predicted RUL'))
        hist_fig.update_layout(
            title="Distribution of Predicted RUL",
            xaxis_title="Predicted RUL",
            yaxis_title="Frequency",
            height=400
        )
        
        return [table_data, table_columns, hist_fig, 
                {'display': 'block'}, {'display': 'block'},
                "✓ Prediction Complete"]
        
    except Exception as e:
        return [[], [], {}, {'display': 'none'}, {'display': 'none'}, 
               f"Error: {str(e)}"]

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)