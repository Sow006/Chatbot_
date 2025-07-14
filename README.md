Overview

This project is a modular AI-powered chatbot built in Python. It uses natural language processing (NLP) and machine learning to recognize user intents, extract entities, and generate appropriate responses. The chatbot supports multi-intent conversation, simple entity extraction, and can integrate with external APIs (example: weather). It features a web interface built with Flask.

Features

Intent recognition using TF-IDF and Logistic Regression
Simple entity extraction (expandable to spaCy)
Multi-turn dialogue management
External API integration (weather example)
Logging and analytics
Web interface using Flask
Easily extensible with new intents and responses

Requirements

Python 3.7+
Flask
scikit-learn
nltk
requests

Install requirements:

bash
pip install flask scikit-learn nltk requests
Setup
Clone or download this repository.

Download NLTK data (first run only):
python
import nltk
nltk.download('punkt')
Create an intents.json file in the project directory (see below for an example).
