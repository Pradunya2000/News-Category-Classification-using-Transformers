from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
