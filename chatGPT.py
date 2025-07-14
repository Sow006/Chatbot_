# chatbot_project.py

import json
import random
import logging
from flask import Flask, request, render_template, jsonify
import requests
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Logging setup
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# 2. Load intents and training data
with open('intents.json') as f:
    intents = json.load(f)

X = []
y = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# 3. NLP Pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

# 4. Intent Recognition
def predict_intent(text):
    return pipeline.predict([text])[0]

# 5. Entity Extraction (simple example)
def extract_entities(text):
    # You can use spaCy for more advanced extraction
    tokens = nltk.word_tokenize(text)
    return [token for token in tokens if token.istitle()]

# 6. API Integration Example (Weather)
def get_weather(city):
    api_key = "YOUR_API_KEY"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        return f"The weather in {city} is {data['current']['condition']['text']} with temperature {data['current']['temp_c']}°C."
    else:
        return "Sorry, I couldn't fetch the weather at the moment."

# 7. Response Generation
def generate_response(user_input):
    intent = predict_intent(user_input)
    logging.info(f"User: {user_input} | Predicted intent: {intent}")
    for i in intents['intents']:
        if i['tag'] == intent:
            if intent == 'weather':
                entities = extract_entities(user_input)
                if entities:
                    return get_weather(entities[0])
                else:
                    return "Please specify a city for the weather."
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

# 8. Flask Web App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # You need to create this HTML file

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
