from typing import Any, List, Optional
from pydantic import BaseModel
import pickle
import pandas as pd
import pycountry_convert as pc
from catboost import CatBoostRegressor
import os

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[Any]


class DataInput(BaseModel):
    job_title: str
    experience_level: str
    employee_country: str
    company_country: str
    # Comentado porque no se usan en el modelo estas variables
    # employment_type: str  # Tipo de empleo (Full_time, Part_time, etc.)
    # remote_ratio: str     # Porcentaje de trabajo remoto
    # company_size: str    # Tamaño de la empresa (Small, Medium, Large)

class MultipleDataInputs(BaseModel):
    inputs: List[DataInput]  # Lista de objetos DataInput
    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "job_title": "Software Engineer",
                        "experience_level": "Mid_level",
                        "employee_country": "United States",
                        "company_country": "United States"
                        # Comentado porque no se usan en el modelo estas variables
                        # ,
                        # "employment_type": "Full_time",
                        # "remote_ratio": "< 20%",
                        # "company_size": "Large (> 250 employees)"
                    }#,
                    # {
                    #     "job_title": "Data Scientist",
                    #     "experience_level": "Senior_level",
                    #     "employee_country": "Canada",
                    #     "company_country": "United States",
                    #     "employment_type": "Contract",
                    #     "remote_ratio": "50-80%",
                    #     "company_size": "Medium (50-250 employees)"
                    # }
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

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model-pkg', 'best_cbr_reg_model_country.pkl')
    
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



