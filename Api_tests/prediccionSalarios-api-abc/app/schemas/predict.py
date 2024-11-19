from typing import Any, List, Optional

from pydantic import BaseModel
#from model.processing.validation import DataInputSchema # Original

class DataInput(BaseModel):
    job_title: Optional[str]
    experience_level: Optional[str]
    employee_country: Optional[str]
    company_country: Optional[str]
    # Comentado porque no se usan en el modelo estas variables
    # employment_type: str  # Tipo de empleo (Full_time, Part_time, etc.)
    # remote_ratio: str     # Porcentaje de trabajo remoto
    # company_size: str    # Tamaño de la empresa (Small, Medium, Large)
    class Config:
        schema_extra = {
            "example": {
                "job_title": "Software Engineer",
                "experience_level": "Mid_level",
                "employee_country": "United States",
                "company_country": "United States"
            }
        }

# Output schema for prediction results
class PredictionResults(BaseModel):
    predictions: Optional[List[float]]  # List of predicted salaries (even for single input)
    errors: Optional[str]  # Errors if any occurred during prediction
    version: str  # Model version or API version
    class Config:
        schema_extra = {
            "example": {
                "predictions": [75000.0],
                "errors": None,
                "version": "1.0.0"
            }
        }
