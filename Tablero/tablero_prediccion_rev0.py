import pickle
import requests
from catboost import CatBoostRegressor
import pycountry_convert as pc
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html

# Load the model
#model_url = "https://raw.githubusercontent.com/juankquintana/prediccion_salarios/main/Models/Classification/best_cb_model_3qua.pkl"
model_url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/branch_alejandra/Models/Classification/best_cbr_reg_model_country.pkl'
response = requests.get(model_url)
response.raise_for_status()  # Ensure the request was successful

# Load the model from the response content
model = pickle.loads(response.content)



def preprocess_inputs(job_title, experience_level, employee_country, company_country):
    # 1. Map experience_level
    experience_map = {
        'Entry_level': 'EN',
        'Mid_level': 'MI',
        'Senior_level': 'SE',
        'Executive_level': 'EX'
    }
    experience_level = experience_map.get(experience_level, experience_level)

    # 2. Convert employee_country to 2-digit country code
    try:
        employee_country = pc.country_name_to_country_alpha2(employee_country)
    except KeyError:
        employee_country = 'Unknown'  # Default to 'Unknown' if country not found

    # 3. Convert company_country to 2-digit country code
    try:
        company_country = pc.country_name_to_country_alpha2(company_country)
    except KeyError:
        company_country = 'Unknown'

    # Return the preprocessed inputs as a DataFrame row
    return pd.DataFrame([{
        'experience_level': experience_level,
        'job_title': job_title,
        'company_country': company_country,
        'employee_country': employee_country
    }])

def predict_salary(job_title, experience_level, employee_country, company_country):
    # Preprocess inputs
    input_data = preprocess_inputs(job_title, experience_level, employee_country, company_country)

    # Predict salary
    salary = model.predict(input_data)
    return salary[0]

# Dashboard

# Incorporate data
#url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/main/Data/data_top10.csv'
url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/branch_alejandra/Data/data_top10_country.csv'
df= pd.read_csv(url)

job_unique_val = sorted(df['job_title'].unique())
exp_unique_val = ['Entry_level', 'Mid_level', 'Senior_level', 'Executive_level']
res_unique_val = sorted(df['employee_country'].unique())
typ_unique_val = ['Full_time', 'Part_time', 'Contractor']
rem_unique_val = ['< 20%', '20% - 80%', '> 80%']
siz_unique_val = ['Small (< 50 employees)', 'Medium (50 - 250 employees)', 'Large (> 250 employees)']
cco_unique_val = sorted(df['company_country'].unique())

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

sidebar = html.Div(
    [
        dbc.Row(
            html.H5('Selecciona las variables', style={'margin-top': '20px', 'margin-left': '20px'}),
            className='bg-primary text-white font-italic'
        ),
        dbc.Row(
            html.Div([
                html.P('Posición', className='font-weight-bold', style={'margin': '12px 0 6px 15px'}),
                dcc.Dropdown(id='my-job-picker', options=[{'label': x, 'value': x} for x in job_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Nivel de Experiencia', className='font-weight-bold', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-exp-picker', options=[{'label': x, 'value': x} for x in exp_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Pais Residencia', className='font-weight-bold', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-res-picker', options=[{'label': x, 'value': x} for x in res_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Tipo de Empleo', className='font-weight-bold', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-typ-picker', options=[{'label': x, 'value': x} for x in typ_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Porcentaje Remoto', className='font-weight-bold', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-rem-picker', options=[{'label': x, 'value': x} for x in rem_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Tamaño Empresa', className='font-weight-bold', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-siz-picker', options=[{'label': x, 'value': x} for x in siz_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.P('Ubicación Empresa', style={'margin': '10px 0 6px 15px'}),
                dcc.Dropdown(id='my-cco-picker', options=[{'label': x, 'value': x} for x in cco_unique_val], style={'width': '225px', 'margin-left': '6px'}),

                html.Button(id='my-button', n_clicks=0, children='Aplicar', style={'width': '225px', 'margin-top': '20px', 'margin-left': '12px'}, className='bg-dark text-white'),
                html.Hr()
            ])
        )
    ]
)

content = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.H5('Predicción Rango Salario', style={'width': '400px', 'margin-top': '20px', 'margin-left': '30px'}),
                className='bg-light'
            ),
        ),
        html.Div(id='output-div', style={'margin-top': '20px', 'margin-left': '30px'})
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9)
            ],
            style={"height": "100vh"}
        ),
    ],
    fluid=True
)

# Update the callback function to collect all seven variables, using only four for prediction
@app.callback(
    Output('output-div', 'children'),
    [Input('my-button', 'n_clicks')],
    [State('my-job-picker', 'value'),
     State('my-exp-picker', 'value'),
     State('my-res-picker', 'value'),
     State('my-cco-picker', 'value'),
     State('my-typ-picker', 'value'),
     State('my-rem-picker', 'value'),
     State('my-siz-picker', 'value')]
)
def update_output(n_clicks, job_title, experience_level, employee_country, company_country, employment_type, remote_ratio, company_size):
    if n_clicks == 0 or None in [job_title, experience_level, employee_country, company_country]:
        return ''
    else:
        # Predict salary using only the required inputs
        prediction = predict_salary(job_title, experience_level, employee_country, company_country)
        
        # Format the prediction to two decimal places
        formatted_prediction = f"{prediction:.2f}"
        
        # Store all inputs for future use if needed
        collected_data = {
            'job_title': job_title,
            'experience_level': experience_level,
            'employee_country': employee_country,
            'company_country': company_country,
            'employment_type': employment_type,
            'remote_ratio': remote_ratio,
            'company_size': company_size
        }

        return f"Predicted Salary Range: {formatted_prediction}. Inputs Collected: {collected_data}"

if __name__ == "__main__":
    app.run_server(debug=True, port=4567)