import json
from typing import Any
import numpy as np
import pandas as pd
from app.schemas.predict import predict_salary
from app.schemas.predict import preprocess_inputs
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
#from model import __version__ as model_version
from app import __version__, schemas
from app.config import settings


api_router = APIRouter()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Predicción usando el modelo cargado.
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info(f"Realizando predicción sobre los inputs: {input_data.inputs}")
    results = predict_salary(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Error de validación de predicción: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Resultados de la predicción: {results.get('predictions')}")
    return results

