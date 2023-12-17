from .preprocess_text import preprocess_text
from .aspect_opinion_extractor import AspectOpinionExtractor
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

@staticmethod
class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.aspect_opinion_extractor = AspectOpinionExtractor()

    def predict(self, review):
        # Preprocess the input review using the existing preprocess_text function
        review = preprocess_text(review)

        # Extract aspects and opinions
        aspects, opinions = self.aspect_opinion_extractor.extract_aspects_and_opinions(review)

        # Prepare input data for the model
        X = [aspect + ' ' + opinion for aspect, opinion in zip(aspects, opinions)]
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=7)

        # Make predictions using the pre-trained model
        predictions = self.model.predict(X_pad)

        # Convert predictions into the desired format
        sentiments = ['positif' if prediction > 0.5 else 'negatif' for prediction in predictions]
        result = [({aspect: opinion}, sentiment) for aspect, opinion, sentiment in zip(aspects, opinions, sentiments)]
        result_json = json.dumps(result)

        return result_json

    def preprocess_input_for_tflite(self, input_string):
        # Preprocess the input review using the existing preprocess_text function
        preprocessed_input = preprocess_text(input_string)

        # Extract aspects and opinions
        aspects, opinions = self.aspect_opinion_extractor.extract_aspects_and_opinions(preprocessed_input)

        # Prepare input data for the TFLite model
        X = [aspect + ' ' + opinion for aspect, opinion in zip(aspects, opinions)]
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=7)

        # Convert the input data to FLOAT32 (required by TFLite)
        X_float32 = np.float32(X_pad)

        # Ensure the input data has the correct shape (n, 7) where n is the number of aspect-opinion pairs
        if len(X_float32.shape) == 3:
            X_float32 = X_float32.reshape((X_float32.shape[1], X_float32.shape[2]))

        if X_float32.shape[1] != 7:
            raise ValueError(f"Invalid shape: {X_float32.shape}. Expected (n, 7).")

        return X_float32