from fastapi import FastAPI
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.models import PredictionRequest, PredictionResponse
from backend.database import create_table, insert_prediction

app = FastAPI(title="News Category Classifier API")

# create DB table on startup
create_table()

MODEL_PATH = "model/classifier"
TOKENIZER_PATH = "model/tokenizer"
LABEL_MAP_PATH = "model/label_map.json"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_category = label_map[str(predicted_index.item())]
    confidence_score = round(confidence.item(), 4)

    # save to SQLite
    insert_prediction(
        input_text=request.text,
        predicted_category=predicted_category,
        confidence=confidence_score
    )

    return {
        "predicted_category": predicted_category,
        "confidence": confidence_score
    }
