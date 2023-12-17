from keras.preprocessing.text import Tokenizer

def create_tokenizer(texts, max_features):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(texts)
    return tokenizer