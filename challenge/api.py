from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from challenge.model import DelayModel

app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for predicting flight delays at SCL airport",
    version="1.0.0"
)

model = DelayModel()


class Flight(BaseModel):
    """Flight data for making delay predictions."""
    OPERA: str
    TIPOVUELO: str
    MES: int

    class Config:
        schema_extra = {
            "example": {
                "OPERA": "Grupo LATAM",
                "TIPOVUELO": "N",
                "MES": 7
            }
        }


class FlightRequest(BaseModel):
    """Request body containing list of flights."""
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def predict(request: FlightRequest) -> Dict[str, List[int]]:
    """Generate delay predictions for a batch of flights."""
    try:
        df = pd.DataFrame([flight.dict() for flight in request.flights])

        if not df['MES'].between(1, 12).all():
            raise HTTPException(status_code=400, detail="Invalid month value")

        if not df['TIPOVUELO'].isin(['N', 'I']).all():
            raise HTTPException(status_code=400, detail="Invalid flight type")

        features = model.preprocess(df)
        predictions = model.predict(features)

        return {"predict": predictions}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))