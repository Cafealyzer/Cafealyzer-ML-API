from keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from helpers.predictor import Predictor
from helpers.tokenizer import create_tokenizer

app = FastAPI(title="Cafealyzer Model-1 API", version="1.0.0")

model_path = 'model3.h5'

model = load_model(model_path)

tokenizer = create_tokenizer([] ,max_features=10000)

predictor = Predictor(model, tokenizer)

@app.get("/")
async def root():
  return {"message": "Hello World"}

class Review(BaseModel):
    reviews: List[str]
    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "cafenya sangat bagus sekali",
                    "pelayanan kurang ramah",
                    "kopinya enak"
                ]
            }
        }

@app.post('/predict-review')
def predict_sentiment(review: Review):
    predictions = predictor.predict(review.reviews)

    return {predictions}