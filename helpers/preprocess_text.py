import json

def preprocess_text(text):
    slang_path = '_json_colloquial-indonesian-lexicon.txt'
    with open(slang_path, 'r') as f:
        slang_dict = json.load(f)
    text = str(text)
    text = text.lower()
    text = text.split()
    text = [slang_dict.get(word, word) for word in text]
    text = ' '.join(text)
    return text