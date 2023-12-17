from keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from helpers.predictor import Predictor
from helpers.tokenizer import create_tokenizer
import tensorflow as tf

app = FastAPI(title="Cafealyzer Model-1 API", version="1.0.0")

model_path = 'absa.h5'

tflite_path = 'modelnon.tflite'

model = load_model(model_path)
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

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
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # predictions = predictor.predict(review.reviews)
    tflite_input_data = predictor.preprocess_input_for_tflite(review.reviews)

    print({"input review: " : review.reviews})

    for i in range(tflite_input_data.shape[0]):
    # Setel tensor input di TFLite Interpreter
      interpreter.set_tensor(input_details[0]['index'], tflite_input_data[i:i+1])

      # Jalankan inference
      interpreter.invoke()

      # Dapatkan tensor output
      output_data = interpreter.get_tensor(output_details[0]['index'])
      print(f"Prediction Result for input {i+1}:", output_data)