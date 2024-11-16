from typing import List, Optional
from pydantic import BaseModel
import pickle
import pandas as pd
import pycountry_convert as pc
from catboost import CatBoostRegressor
import os
 

# Definir los esquemas de entrada y salida para la API
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[Any[float]]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]  # Aquí asumo que DataInputSchema define el formato de los datos individuales
    # Campos adicionales que el modelo necesita según tu código:
    job_title: str
    experience_level: str
    employee_country: str
    company_country: str
    employment_type: str  # Tipo de empleo (Full_time, Part_time, etc.)
    remote_ratio: str     # Porcentaje de trabajo remoto
    company_size: str    # Tamaño de la empresa (Small, Medium, Large)
    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "job_title": "Software Engineer",
                        "experience_level": "Mid_level",
                        "employee_country": "United States",
                        "company_country": "United States",
                        "employment_type": "Full_time",
                        "remote_ratio": "< 20%",
                        "company_size": "Large (> 250 employees)"
                    }
                ]
            }
        }


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

# Defir función para cargar el modelo
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model-pkg', 'best_cbr_reg_model_country.pkl')
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



