import pickle
import requests
#from catboost import CatBoostRegressor
import pycountry_convert as pc
import pandas as pd
import numpy as np

#import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html, Dash

import plotly.express as px
import plotly.colors

# Load data
#url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/main/Data/data_top10.csv'
url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/refs/heads/branch_alejandra/Data/data_top10_country.csv'
df= pd.read_csv(url)

# Load model
#model_url = "https://raw.githubusercontent.com/juankquintana/prediccion_salarios/main/Models/Classification/best_cb_model_3qua.pkl"
model_url = 'https://raw.githubusercontent.com/juankquintana/prediccion_salarios/branch_alejandra/Models/Classification/best_cbr_reg_model_country.pkl'
response = requests.get(model_url)
response.raise_for_status()  # Ensure the request was successful

model = pickle.loads(response.content)

# Input Processing
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

# Salary - Ranges 
max_salary = df['salary_in_usd'].max()
min_salary = df['salary_in_usd'].min()

num_subranges = 25
subranges = np.linspace(min_salary, max_salary, num=num_subranges+1, endpoint=True)
ranges = []
for i in range(len(subranges) - 1):
    subrange_min = int(subranges[i])
    subrange_max = int(subranges[i + 1])
    # Append as a tuple (subrange_min, subrange_max) instead of a formatted string
    ranges.append((subrange_min, subrange_max))

# Prediction
def predict_salary_range(job_title, experience_level, employee_country, company_country):
    # Preprocess inputs
    input_data = preprocess_inputs(job_title, experience_level, employee_country, company_country)

    # Predict salary
    salary = model.predict(input_data)
    prediction_range = None
    for range_min, range_max in ranges:
        if range_min <= salary < range_max:
            prediction_range = f"{range_min:,} - {range_max:,}"
            break
    return  prediction_range,  #salary[0],


# Dashboard
job_unique_val = sorted(df['job_title'].unique())
exp_unique_val = ['Entry_level', 'Mid_level', 'Senior_level', 'Executive_level']
res_unique_val = sorted(df['employee_country'].unique())
typ_unique_val = ['Full_time', 'Part_time', 'Contractor']
rem_unique_val = ['< 20%', '20% - 80%', '> 80%']
siz_unique_val = ['Small (< 50 employees)', 'Medium (50 - 250 employees)', 'Large (> 250 employees)']
cco_unique_val = sorted(df['company_country'].unique())

app = Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        external_stylesheets=[dbc.themes.FLATLY]
        )
app.title = "Tablero Predicción de Salarios"

server = app.server # Not sure what does this do (app.py W1)
app.config.suppress_callback_exceptions = True

sidebar = html.Div(
    [
        dbc.Row(
            html.H5('Selecciona las variables', style={'margin-top': '20px', 'margin-left': '20px'}),
            className='bg-primary text-white font-italic'
        ),
        
        # Add a row for all dropdowns and labels, with even spacing
        dbc.Row(
            html.Div([
                dbc.Row([
                    dbc.Col(html.P('Posición', className='font-weight-bold', style={'margin': '12px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-job-picker', options=[{'label': x, 'value': x} for x in job_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Nivel de Experiencia', className='font-weight-bold', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-exp-picker', options=[{'label': x, 'value': x} for x in exp_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Pais Residencia', className='font-weight-bold', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-res-picker', options=[{'label': x, 'value': x} for x in res_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Tipo de Empleo', className='font-weight-bold', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-typ-picker', options=[{'label': x, 'value': x} for x in typ_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Porcentaje Remoto', className='font-weight-bold', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-rem-picker', options=[{'label': x, 'value': x} for x in rem_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Tamaño Empresa', className='font-weight-bold', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-siz-picker', options=[{'label': x, 'value': x} for x in siz_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                dbc.Row([
                    dbc.Col(html.P('Ubicación Empresa', style={'margin': '10px 0 6px 15px'})),
                    dbc.Col(dcc.Dropdown(id='my-cco-picker', options=[{'label': x, 'value': x} for x in cco_unique_val], style={'width': '222px', 'margin-left': '6px'})),
                ], style={"height": "10vh"}),

                # Apply button, aligned with the other dropdowns
                dbc.Row([
                    dbc.Col(html.Button(id='my-button', n_clicks=0, children='Aplicar', style={'width': '222px', 'margin-top': '20px', 'margin-left': '12px'}, className='bg-dark text-white')),
                ], style={"height": "10vh"})
            ])
        )
    ]
)

content = html.Div(
    [
        dcc.Store(id='figures-visible', data=False),
        
        dbc.Row(
            dbc.Col(
                html.H5('Predicción Rango Salario', style={'width': '400px', 'margin-top': '20px', 'margin-left': '30px'}),
                className='bg-light'
            ),
        ),

        dbc.Row(
            dbc.Col(
                html.Div(id='output-div', style={'margin-top': '10px', 'margin-left': '30px'}),
                width=12
            )
        ),

        dbc.Row(
            dbc.Col(dcc.Graph(id='map-fig',
                               style={'height': '300px', 'margin-left': '10px'}
                               ),
                                width=12),
            justify="left"
        ),

        dbc.Row(
            dbc.Col(dcc.Graph(id='bar-fig',
                               style={'height': '300px','margin-left': '20px'}
                               ),
                                width=12),
            #justify="left"
        ),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9)
            ],
            style={"height": "80vh"}
        ),
    ],
    fluid=True
)

# Update the callback function 
@app.callback(
    [Output('map-fig', 'figure'),
     Output('bar-fig', 'figure'),
     Output('output-div', 'children')],
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
        return {}, {}, ''  
    else:
        prediction_range = predict_salary_range(job_title, experience_level, employee_country, company_country)
        
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
        formatted_output = f"Para las variables seleccionadas el rango de salario esperado en USD es:  {prediction_range[0]}"
        
        output_card =  dbc.Card(
            dbc.CardBody(
                dcc.Markdown(formatted_output, style={
                    'fontSize': '18px',  
                    'color': '#440154'   
                })
            ),
            style={
                'margin-top': '1px', 
                'padding': '3px', 
                'backgroundColor': '#5EB2BE',  
                'border': '2px solid #440154',  
                'borderRadius': '5px',
                'height': '50px'
            }
        )
        # Filter data for the selected position
        filtered_df = df[df['job_title'] == job_title]
        
        # Update map figure
        country_median_df = filtered_df.groupby("employee_country", as_index=False).median(numeric_only=True)
        map_fig = px.choropleth(
            country_median_df,
            locations="employee_country",
            locationmode="country names",
            color="salary_in_usd",
            hover_name="employee_country",
            color_continuous_scale="Viridis",
            title=f"    Salario promedio por país - Posición : {job_title}",
        )
        map_fig.update_geos(
            projection_type="robinson",
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            showcountries=True,
            countrycolor="Black"
        )
        map_fig.update_layout(
            margin={"r":0, "t":50, "l":0, "b":0},
            height=300  
        ) 
        # Update bar figure
        region_median_df = filtered_df.groupby("employee_residence", as_index=False).median(numeric_only=True)
        bar_fig = px.bar(
            region_median_df,
            x="employee_residence",
            y="salary_in_usd",
            text="salary_in_usd",
            title=f"  Salario promedio por continente - Posición: {job_title}"
        )
        
        # Apply Viridis color scale to bar chart
        color_scale = plotly.colors.sequential.Viridis
        min_salary = region_median_df['salary_in_usd'].min()
        max_salary = region_median_df['salary_in_usd'].max()
        region_median_df['color'] = region_median_df['salary_in_usd'].apply(
            lambda x: color_scale[int((x - min_salary) / (max_salary - min_salary) * (len(color_scale) - 1))]
        )
        bar_fig.update_traces(marker_color=region_median_df['color'])
        bar_fig.update_layout(
            margin={"r":0, "t":50, "l":0, "b":0},
            height=300 
        )
        # Return map_fig, bar_fig, and the output card
        return map_fig, bar_fig, output_card

if __name__ == "__main__":
    app.run_server(debug=True, port=5678)