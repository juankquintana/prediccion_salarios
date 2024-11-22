import pickle
import pandas as pd
import pycountry_convert as pc
import os


# Definir funciones de predicción
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

def load_model():
    #model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model-pkg', 'best_cbr_reg_model_country.pkl') # original
    model_path = os.path.join(os.path.dirname(__file__), 'best_cbr_reg_model_country_api.pkl') # Para correrlo localmente
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Cargar el modelo
model = load_model()

def predict_salary(job_title, experience_level, employee_country, company_country):
    # Preprocess inputs
    input_data = preprocess_inputs(job_title, experience_level, employee_country, company_country)

    # Predict salary
    salary = model.predict(input_data)
    return salary[0]