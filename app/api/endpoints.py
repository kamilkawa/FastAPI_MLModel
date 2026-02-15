from fastapi import APIRouter
from app.schemas.iris import IrisInput, IrisOutput
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()


@router.post("/predict", response_model=IrisOutput)
def predict_species(data: IrisInput) -> IrisOutput:
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    prediction = model_service.predict(features)
    return IrisOutput(species=prediction)
