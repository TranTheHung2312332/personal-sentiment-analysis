from fastapi import APIRouter
from services.predict import predict as service__predict
from schemas.predict import PredictResponse, PredictRequest
from fastapi import HTTPException

router = APIRouter(
    prefix="/predict",
    tags=["Inference"]
)

@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return service__predict(req)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )