import dash
import dash_bootstrap_components as dbc
import requests
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Función para hacer la solicitud a la API
def get_prediction(job_title, experience_level, employee_country, company_country):
    response = requests.post(
        'http://localhost:8000/predict',  # Suponiendo que la API corre en localhost
        json={
            'job_title': job_title,
            'experience_level': experience_level,
            'employee_country': employee_country,
            'company_country': company_country
        }
    )
    response.raise_for_status()
    return response.json()['predicted_salary']

# Aquí va el resto del código del dashboard Dash
@app.callback(
    Output('output-div', 'children'),
    [Input('my-button', 'n_clicks')],
    [State('my-job-picker', 'value'),
     State('my-exp-picker', 'value'),
     State('my-res-picker', 'value'),
     State('my-cco-picker', 'value')]
)
def update_output(n_clicks, job_title, experience_level, employee_country, company_country):
    if n_clicks == 0 or None in [job_title, experience_level, employee_country, company_country]:
        return ''
    else:
        # Obtener la predicción llamando a la API
        prediction = get_prediction(job_title, experience_level, employee_country, company_country)
        
        return f"Predicted Salary: {prediction:.2f}"

if __name__ == "__main__":
    app.run_server(debug=True, port=4567)
