from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Initialize Flask app
app = Flask(__name__)

# Load the necessary files
lemmatizer = WordNetLemmatizer()
model = load_model(r'E:\Edunet Internship\my_chatbot\chatbot_model.h5')
intents = json.loads(open(r'E:\Edunet Internship\my_chatbot\intents.json').read())
words = pickle.load(open(r'E:\Edunet Internship\my_chatbot\words.pkl', 'rb'))
classes = pickle.load(open(r'E:\Edunet Internship\my_chatbot\classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    """
    Tokenize and lemmatize the input sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """
    Create a bag of words for the input sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """
    Predict the class of the input sentence.
    """
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return []
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    """
    Get the response for the predicted intent.
    Handles cases where no intent is matched.
    """
    if not ints:  # Handle case with no matching intents
        return "I'm not sure I understand that. Can you try rephrasing?"
    
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand that. Can you try rephrasing?"

@app.route('/')
def home():
    """
    Render the chatbot home page.
    Ensure there is an 'index.html' file in the templates folder.
    """
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def chatbot_response():
    """
    Process the user message and return a response.
    """
    try:
        msg = request.args.get('msg')
        if not msg:
            return "Please provide a valid input."
        
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
        return res
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
