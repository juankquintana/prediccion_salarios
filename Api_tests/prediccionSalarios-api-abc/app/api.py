import json
from typing import Any

from model.make_prediction import predict_salary 
from app.schemas import DataInput, PredictionResults

from fastapi import APIRouter, HTTPException
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from loguru import logger

from app import __version__, schemas
from app.config import settings

model_version = "1.0.0" # Added this to fix error in Health schema

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


# Ruta para realizar las predicciones
@api_router.post(
        "/predict",
        response_model=schemas.PredictionResults,
        status_code=200,
        summary="Predict Salary",
    description="Predicts the salary based on job details.",
)

async def predict(input_data: DataInput):
    """
    Predict endpoint for a single input.
    """
    try:
        # Call the prediction function
        prediction = predict_salary(
            job_title=input_data.job_title,
            experience_level=input_data.experience_level,
            employee_country=input_data.employee_country,
            company_country=input_data.company_country
        )

        # Return the prediction
        return PredictionResults(
            predictions=[prediction],  # Wrap in a list to maintain compatibility
            errors=None,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# async def predict(input_data: schemas.MultipleDataInputs) -> Any: # ORIGINAL
#async def predict(request: Request, input_data: DataInput):
    # Log the raw request body
    #raw_body = await request.body()
    #logger.info(f"Raw request body: {raw_body.decode('utf-8')}")


    #logger.info(f"Realizando predicción sobre los inputs: {input_data.inputs}") # ORIGINAL
    #logger.info(f"Realizando predicción sobre los inputs: {input_data.dict()}") # Cambie orden
    #input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    #results = predict_salary(input_data=input_df.replace({np.nan: None}))

    #if results["errors"] is not None:
        #logger.warning(f"Error de validación de predicción: {results.get('errors')}")
        #raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    #logger.info(f"Resultados de la predicción: {results.get('predictions')}")
    #return results
