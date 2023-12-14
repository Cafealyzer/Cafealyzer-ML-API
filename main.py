from helpers.text_processing import text_cleaning
from keras.models import load_model
from fastapi import FastAPI
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

app = FastAPI(title="Cafealyzer Model-1 API", version="1.0.0")

model_path = 'ABSA-CNN.h5'

model = load_model(model_path)

@app.get("/")
async def root():
  print(model)
  return {"message": "Hello World"}

# @app.get("/predict-review")
# def predict_sentiment(review: str):
    
#     cleaned_review = text_cleaning(review)
    # print(review + ' original')
    # print(cleaned_review + ' cleaned')
    
    # prediction = model.predict([cleaned_review])
    # print(prediction)
    # output = int(prediction[0])
    # probas = model.predict_proba([cleaned_review])
    # output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # sentiments = {0: "Negative", 1: "Positive"}
  
    # result = {"prediction": sentiments[output], "Probability": output_probability}
    # return result
    

@app.get("/predict-review")
def predict_sentiment(review: str):
    # Bersihkan review
    cleaned_review = text_cleaning(review)

    # Ubah review menjadi sequence of integers
    tokenizer = Tokenizer()
    review_sequence = tokenizer.texts_to_sequences([cleaned_review])

    # Lakukan padding pada sequence
    review_padded = pad_sequences(review_sequence, maxlen=100)

    # Prediksi sentimen review
    prediction = model.predict(review_padded)

    # Kembalikan hasil prediksi
    return {"prediction": prediction}